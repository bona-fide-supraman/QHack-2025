// ESP32 + MAX9814 (analog mic) -> I2S-ADC -> FFT -> 5 parallel WS2812 strips
// Strips: lengths {6, 5, 5, 5, 5}, each DIN on its OWN ESP32 pin.
// Board: ESP32 Dev Module
// Libraries: Adafruit NeoPixel, arduinoFFT (v2.x)

#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include <arduinoFFT.h>
#include <driver/i2s.h>
#include <driver/adc.h>
#include <math.h>

// ------------------- USER CONFIG -------------------
const uint8_t NUM_STRIPS = 5;
const uint8_t DATA_PINS[NUM_STRIPS]  = {18, 19, 21, 22, 23}; // DIN pins (one per strip)
const uint16_t STRIP_LEN[NUM_STRIPS] = {6, 5, 5, 5, 5};      // LEDs per strip
const uint8_t BRIGHTNESS = 80;                               // 0..255
const bool SERPENTINE = false; // true if alternate strips are physically flipped

// MAX9814 wiring: VDD->3V3, GND->GND, OUT->GPIO34
#define ADC_GPIO      34                   // ADC1_CH6
#define ADC_CHANNEL   ADC1_CHANNEL_6
#define SAMPLE_RATE   1500
#define SAMPLES       1024                 // power of two
#define I2S_PORT      I2S_NUM_0
// ---------------------------------------------------

Adafruit_NeoPixel* strips[NUM_STRIPS];

// FFT buffers + object (arduinoFFT v2.x)
double vReal[SAMPLES];
double vImag[SAMPLES];
ArduinoFFT<double> FFT = ArduinoFFT<double>(vReal, vImag, SAMPLES, SAMPLE_RATE);

// 5 bands (Hz) -> map each to a strip
const double bandEdgesHz[6] = { 80, 250, 500, 1000, 2000, 4000 };

// smoothing + simple auto-gain
float bandEMA[NUM_STRIPS] = {0,0,0,0,0};
const float emaAttack = 0.40f, emaDecay = 0.12f;
float gain = 1.0f;
const float targetLevel = 0.70f; // tallest bar ~70% height
const float gainAttack = 0.010f, gainDecay = 0.001f;

// ---------- LED HELPERS ----------
inline uint32_t rgb(uint8_t r, uint8_t g, uint8_t b) { return strips[0]->Color(r, g, b); }

void setPixel(uint8_t stripIdx, uint16_t pixelIdx, uint32_t c) {
  if (stripIdx >= NUM_STRIPS) return;
  if (pixelIdx >= STRIP_LEN[stripIdx]) return;
  strips[stripIdx]->setPixelColor(pixelIdx, c);
}

void setXY(uint8_t x, uint16_t y, uint32_t c) {
  if (x >= NUM_STRIPS) return;
  uint16_t len = STRIP_LEN[x];
  if (y >= len) return;
  uint16_t idx = (SERPENTINE && (x % 2 == 1)) ? (len - 1 - y) : y;
  strips[x]->setPixelColor(idx, c);
}

void clearAll(bool showNow = true) {
  for (uint8_t s = 0; s < NUM_STRIPS; s++) {
    for (uint16_t i = 0; i < STRIP_LEN[s]; i++) strips[s]->setPixelColor(i, 0);
    if (showNow) strips[s]->show();
  }
}
void showAll() { for (uint8_t s=0; s<NUM_STRIPS; s++) strips[s]->show(); }

uint32_t heatColor(uint8_t level, uint8_t maxLevel) {
  float t = (float)level / (float)maxLevel; if (t < 0) t = 0; if (t > 1) t = 1;
  uint8_t r = (uint8_t)( (t * 1.6f > 1.0f ? 1.0f : t * 1.6f) * 255.0f );
  uint8_t g = (uint8_t)( (t < 0.5f ? (t*2.0f) : (1.0f - (t-0.5f)*2.0f)) * 255.0f );
  return strips[0]->Color(r, g, 0);
}

void drawBars(const uint16_t heights[NUM_STRIPS]) {
  for (uint8_t x=0; x<NUM_STRIPS; x++) {
    for (uint16_t y=0; y<STRIP_LEN[x]; y++) {
      uint32_t c = (y < heights[x]) ? heatColor(y+1, STRIP_LEN[x]) : 0;
      setXY(x, y, c);
    }
  }
  for (uint8_t s=0; s<NUM_STRIPS; s++) strips[s]->setBrightness(BRIGHTNESS);
  showAll();
}

void setAllRedHalf() {
  for (uint8_t s=0; s<NUM_STRIPS; s++) {
    strips[s]->setBrightness(128); // half
    for (uint16_t i=0; i<STRIP_LEN[s]; i++) strips[s]->setPixelColor(i, rgb(128,0,0));
    strips[s]->show();
  }
}

// ---------- I2S-ADC (sampling analog mic via ADC1) ----------
void setupI2SADC() {
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(ADC_CHANNEL, ADC_ATTEN_DB_11); // ~3.3V FS

  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S_MSB,
    .intr_alloc_flags = 0,
    .dma_buf_count = 4,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
  i2s_set_adc_mode(ADC_UNIT_1, ADC_CHANNEL);
  i2s_set_clk(I2S_PORT, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_16BIT, I2S_CHANNEL_MONO);
  i2s_adc_enable(I2S_PORT);
}

// Small sanity check for mic: print mean / acRMS / peaks for ~2s
void micSanityCheck(uint32_t ms = 2000) {
  Serial.println("Mic sanity check (mean/acRMS/lo/hi)...");
  uint32_t t0 = millis();
  static uint16_t raw[SAMPLES];
  while (millis() - t0 < ms) {
    size_t br = 0;
    i2s_read(I2S_PORT, (void*)raw, sizeof(raw), &br, portMAX_DELAY);
    int n = br / sizeof(uint16_t);
    if (n <= 0) continue;

    uint32_t sum = 0, sumsq = 0;
    uint16_t lo = 4095, hi = 0;
    for (int i=0; i<n; i++) {
      uint16_t v = raw[i] & 0x0FFF; // 12-bit
      sum += v; sumsq += (uint32_t)v * v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    float mean = sum / (float)n;
    float rms  = sqrtf(sumsq / (float)n);
    float acRms = sqrtf(fmaxf(0.0f, rms*rms - mean*mean)); // <-- variable is 'acRms'
    Serial.printf("mean=%4.0f  acRMS=%5.1f  lo=%4u  hi=%4u\n", mean, acRms, lo, hi);
    delay(100);
  }
  Serial.println("Mic check done.");
}

void setup() {
  // LEDs
  for (uint8_t s=0; s<NUM_STRIPS; s++) {
    strips[s] = new Adafruit_NeoPixel(STRIP_LEN[s], DATA_PINS[s], NEO_GRB + NEO_KHZ800);
    strips[s]->begin();
    strips[s]->setBrightness(BRIGHTNESS);
    strips[s]->show();
  }

  Serial.begin(115200);
  setupI2SADC();

  // Visual sanity + mic check
  setAllRedHalf();
  delay(500);
  clearAll();
  micSanityCheck(2000);
}

void loop() {
  // 1) Get a block of samples from I2S ADC
  static uint16_t raw[SAMPLES];
  size_t bytesRead = 0;
  i2s_read(I2S_PORT, (void*)raw, sizeof(raw), &bytesRead, portMAX_DELAY);
  if (bytesRead < sizeof(raw)) return;

  // 2) Prepare FFT input: remove DC (no manual Hann here; FFT will window)
  double dc = 0;
  for (int i=0; i<SAMPLES; i++) { vReal[i] = (double)(raw[i] & 0x0FFF); dc += vReal[i]; }
  dc /= SAMPLES;
  for (int i=0; i<SAMPLES; i++) { vReal[i] -= dc; vImag[i] = 0.0; }

  // 3) FFT (arduinoFFT v2.x)
  FFT.windowing(FFTWindow::Hann, FFTDirection::Forward);
  FFT.compute(FFTDirection::Forward);
  FFT.complexToMagnitude();

  // 4) Average bins into 5 bands
  const double fRes = (double)SAMPLE_RATE / (double)SAMPLES;
  float bandVals[NUM_STRIPS] = {0,0,0,0,0};
  for (int b=0; b<NUM_STRIPS; b++) {
    int k0 = max(1, (int)lround(bandEdgesHz[b]   / fRes));
    int k1 = min(SAMPLES/2 - 1, (int)lround(bandEdgesHz[b+1] / fRes));
    double sum = 0;
    for (int k=k0; k<=k1; k++) sum += vReal[k];
    bandVals[b] = (float)(sum / max(1, (k1 - k0 + 1)));
  }

  // 5) Auto-gain so peak approaches targetLevel
  float peak = 0;
  for (int b=0; b<NUM_STRIPS; b++) peak = max(peak, bandVals[b] * gain);
  float err = targetLevel - peak;
  gain *= (1.0f + (err > 0 ? gainAttack : gainDecay) * err);
  gain = constrain(gain, 0.05f, 50.0f);

  // 6) Smooth + compress -> heights
  uint16_t heights[NUM_STRIPS];
  for (int b=0; b<NUM_STRIPS; b++) {
    float v = bandVals[b] * gain;
    float ema = (v > bandEMA[b]) ? (emaAttack*v + (1-emaAttack)*bandEMA[b])
                                 : (emaDecay *v + (1-emaDecay )*bandEMA[b]);
    bandEMA[b] = ema;
    float level01 = log10f(1.0f + 9.0f * ema); // 0..~1
    int h = (int)round(level01 * STRIP_LEN[b]);
    heights[b] = (uint16_t) constrain(h, 0, (int)STRIP_LEN[b]);
  }

  // 7) Draw bars
  drawBars(heights);
}
