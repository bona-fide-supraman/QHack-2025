// ESP32 + MAX9814 -> 300-sample plain DFT -> 5 parallel WS2812 bars (robust continuous)
//
// MAX9814: VDD->3V3, GND->GND, OUT->GPIO34
// Strips {6,5,5,5,5} on GPIOs {18,19,21,22,23}; each DIN via 330–470Ω; LED 5V supply with 1000µF cap; COMMON GND.
//
// Board: ESP32 Dev Module
// Libs: Adafruit NeoPixel

#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include <math.h>

// ---------- CONFIG ----------
#define MIC_PIN   34
#define FS_HZ     4000          // sample rate (Hz)
#define N         300           // samples per frame

const uint8_t NUM_STRIPS = 5;
const uint8_t DATA_PINS[NUM_STRIPS]  = {18, 19, 21, 22, 23};
const uint16_t STRIP_LEN[NUM_STRIPS] = {6, 5, 5, 5, 5};
const uint8_t BRIGHTNESS = 35;       // modest; raise later if stable
const bool SERPENTINE = false;

#define HEARTBEAT_PIN 2         // onboard LED on many ESP32 dev boards

// Speech-friendly bands; start above rumble to keep bar 0 lively
const float BAND_EDGE_HZ[6] = { 120, 300, 700, 1200, 2000, (FS_HZ/2.0f) };

// Sensitivity / smoothing / AGC
const float MIC_BOOST   = 2.0f;
const float NOISE0      = 0.0045f;   // higher gate for band 0 (kills hum/AC rumble)
const float NOISE_OTH   = 0.0025f;
const float TARGET_LVL  = 0.75f;     // target peak after compression
const float AGC_ALPHA   = 0.02f;     // AGC speed
const float EMA_ATTACK  = 0.60f;     // bar rise
const float EMA_DECAY   = 0.18f;     // bar fall

Adafruit_NeoPixel* strips[NUM_STRIPS];

static float x[N];                 // time samples (DC-removed)
static float mag[N/2 + 1];         // magnitude spectrum
static float ema[NUM_STRIPS] = {0,0,0,0,0};
static float agc = 2.0f;           // start a bit hot

// ---------- LED helpers ----------
inline uint32_t heatColor(uint8_t level, uint8_t maxLevel) {
  float t = (float)level / (float)maxLevel; t = constrain(t, 0, 1);
  uint8_t r = (uint8_t)(min(1.0f, t*1.6f)*255);
  uint8_t g = (uint8_t)(((t<0.5f)? t*2.0f : 1.0f-(t-0.5f)*2.0f)*255);
  return strips[0]->Color(r, g, 0);
}
void setBar(uint8_t s, uint16_t h) {
  uint16_t len = STRIP_LEN[s]; if (h > len) h = len;
  for (uint16_t y=0; y<len; y++) {
    uint32_t c = (y < h) ? heatColor(y+1, len) : 0;
    uint16_t idx = (SERPENTINE && (s & 1)) ? (len-1-y) : y;
    strips[s]->setPixelColor(idx, c);
  }
}
void showAll() {
  for (uint8_t s=0; s<NUM_STRIPS; s++) { strips[s]->setBrightness(BRIGHTNESS); strips[s]->show(); }
}

// ---------- Sampling (timed; watchdog-friendly) ----------
void sampleBlock() {
  const uint32_t Ts_us = 1000000UL / FS_HZ;
  uint32_t next = micros();
  uint32_t sum = 0;
  for (int i=0; i<N; i++) {
    // sleep until next tick (keeps CPU cooler, avoids starving RTOS)
    uint32_t now = micros();
    int32_t dt = (int32_t)(next - now);
    if (dt > 4) delayMicroseconds(dt - 3);    // rough align
    while ((int32_t)(micros() - next) < 0) { /* fine wait */ }
    next += Ts_us;

    uint16_t v = analogRead(MIC_PIN);  // 0..4095
    sum += v;
    x[i] = (float)v;

    if ((i & 31) == 0) yield();        // feed watchdog
  }
  float mean = (float)sum / N;
  for (int i=0; i<N; i++) x[i] -= mean; // DC remove
}

// ---------- Plain DFT (iterative twiddle; watchdog-friendly) ----------
void dftBasic() {
  const int K = N/2;
  float re0=0; for (int n=0; n<N; n++) re0 += x[n];
  mag[0] = fabsf(re0);

  for (int k=1; k<=K; k++) {
    const float theta = -2.0f * (float)M_PI * k / (float)N;
    const float cs = cosf(theta), sn = sinf(theta);
    float c=1.0f, s=0.0f, re=0.0f, im=0.0f;
    for (int n=0; n<N; n++) {
      float xn = x[n];
      re += xn * c;
      im += xn * s;
      float c_new = c*cs - s*sn;
      s = c*sn + s*cs;
      c = c_new;
      if ((n & 63) == 0) yield();  // inner-loop yield
    }
    mag[k] = sqrtf(re*re + im*im);
    if ((k & 7) == 0) yield();     // outer-loop yield
  }
}

void setup() {
  pinMode(HEARTBEAT_PIN, OUTPUT);
  digitalWrite(HEARTBEAT_PIN, LOW);

  Serial.begin(115200);
  analogReadResolution(12);
  analogSetPinAttenuation(MIC_PIN, ADC_11db);    // ~3.3V FS (MAX9814 @3.3V)

  for (uint8_t s=0; s<NUM_STRIPS; s++) {
    strips[s] = new Adafruit_NeoPixel(STRIP_LEN[s], DATA_PINS[s], NEO_GRB + NEO_KHZ800);
    strips[s]->begin();
    strips[s]->setBrightness(BRIGHTNESS);
    strips[s]->clear();
    strips[s]->show();
  }
}

void loop() {
  static uint32_t beatT = 0;
  // 1) Sample continuously
  sampleBlock();

  // 2) Transform
  dftBasic();

  // 3) Collapse to 5 bands with gentle gating & EMA
  const float fRes = (float)FS_HZ / (float)N;
  float bands[NUM_STRIPS] = {0,0,0,0,0};
  for (int b=0; b<NUM_STRIPS; b++) {
    int k0 = (int)ceilf(BAND_EDGE_HZ[b]    / fRes); if (k0 < 1) k0 = 1;
    int k1 = (int)floorf(BAND_EDGE_HZ[b+1] / fRes); int K = N/2; if (k1 > K) k1 = K;

    float v = 0.0f; int cnt = max(1, k1 - k0 + 1);
    for (int k=k0; k<=k1; k++) v += mag[k];
    v = (v / cnt) * MIC_BOOST * agc;

    float gate = (b == 0) ? NOISE0 : NOISE_OTH;
    v = (v > gate) ? (v - gate) : 0.0f;

    // smooth
    ema[b] = (v > ema[b]) ? (EMA_ATTACK*v + (1-EMA_ATTACK)*ema[b])
                          : (EMA_DECAY *v + (1-EMA_DECAY )*ema[b]);
    bands[b] = ema[b];
  }

  // 4) AGC from speech bands (1..4), ignore band 0 so it doesn't pin gain
  float peak = 0; for (int b=1; b<NUM_STRIPS; b++) if (bands[b] > peak) peak = bands[b];
  float peak01 = log10f(1.0f + 9.0f * peak);
  agc *= (1.0f + AGC_ALPHA * (TARGET_LVL - peak01));
  agc = constrain(agc, 0.05f, 30.0f);

  // Safety: if everything is near-zero for a long time, re-center AGC
  static uint32_t quietFrames = 0;
  if (peak01 < 0.02f) { if (++quietFrames > 300) { agc = 2.0f; quietFrames = 0; } }
  else quietFrames = 0;

  // 5) Draw bars
  for (int b=0; b<NUM_STRIPS; b++) {
    float lvl01 = log10f(1.0f + 9.0f * bands[b]);
    uint16_t h = (uint16_t)roundf(lvl01 * STRIP_LEN[b]);
    setBar(b, h);
  }
  showAll();

  // Heartbeat: blinks every ~250 ms to prove the loop is alive
  if (millis() - beatT > 250) { beatT = millis(); digitalWrite(HEARTBEAT_PIN, !digitalRead(HEARTBEAT_PIN)); }

  delay(1); // tiny breather for RTOS
}
