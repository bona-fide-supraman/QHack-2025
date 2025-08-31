// ESP32 + 5 parallel WS2812 strips: lengths {6,5,5,5,5}
// Sets ALL LEDs solid red at half brightness.

#include <Adafruit_NeoPixel.h>

const uint8_t NUM_STRIPS = 5;
const uint8_t DATA_PINS[NUM_STRIPS]  = {18, 19, 21, 22, 23}; // one GPIO per strip DIN
const uint16_t STRIP_LEN[NUM_STRIPS] = {6, 5, 5, 5, 5};

Adafruit_NeoPixel* strips[NUM_STRIPS];

void setup() {
  for (uint8_t s = 0; s < NUM_STRIPS; s++) {
    strips[s] = new Adafruit_NeoPixel(STRIP_LEN[s], DATA_PINS[s], NEO_GRB + NEO_KHZ800);
    strips[s]->begin();
    strips[s]->setBrightness(128); // half of 255
    for (uint16_t i = 0; i < STRIP_LEN[s]; i++) {
      strips[s]->setPixelColor(i, strips[s]->Color(128, 0, 0)); // solid red
    }
    strips[s]->show();
  }
}

void loop() { /* nothing */ }
