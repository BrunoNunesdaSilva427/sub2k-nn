
#include <avr/pgmspace.h>
#include "nn_table.h"

#define SYNC_BYTE 0xAA
#define SERIAL_TIMEOUT_MS 2000

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(SERIAL_TIMEOUT_MS);
  Serial.println(F("sub2k-nn pronto. Envie [0xAA][64 pixels][checksum]."));
  Serial.print(F("Tabela: "));
  Serial.print(NN_INPUT_SIZE);
  Serial.print(F(" entradas, "));
  Serial.print(NN_HIDDEN_SIZE);
  Serial.print(F(" ocultos, "));
  Serial.print(NN_NUM_CLASSES);
  Serial.println(F(" classes."));
}


int8_t infer(const uint8_t* pixels, int32_t* outScores) {
  int32_t hidden[NN_HIDDEN_SIZE];

  
  for (uint8_t j = 0; j < NN_HIDDEN_SIZE; j++) {
    int32_t acc = (int32_t)pgm_read_dword(&NN_B1[j]);
    for (uint8_t i = 0; i < NN_INPUT_SIZE; i++) {
      int8_t w = (int8_t)pgm_read_byte(&NN_W1[i][j]);
      acc += (int32_t)pixels[i] * w;
    }
    hidden[j] = acc > 0 ? acc : 0; 
  }

  
  int8_t bestClass = 0;
  int32_t bestScore = -2147483647L;
  for (uint8_t k = 0; k < NN_NUM_CLASSES; k++) {
    int32_t acc = (int32_t)pgm_read_dword(&NN_B2[k]);
    for (uint8_t j = 0; j < NN_HIDDEN_SIZE; j++) {
      int8_t w = (int8_t)pgm_read_byte(&NN_W2[j][k]);
      acc += hidden[j] * w;
    }
    outScores[k] = acc;
    if (acc > bestScore) {
      bestScore = acc;
      bestClass = k;
    }
  }

  return bestClass;
}

void loop() {
  if (Serial.available() <= 0) return;
  if (Serial.peek() != SYNC_BYTE) {
    Serial.read(); 
    return;
  }
  Serial.read(); 

  uint8_t pixels[NN_INPUT_SIZE];
  size_t got = Serial.readBytes((char*)pixels, NN_INPUT_SIZE);
  if (got != NN_INPUT_SIZE) {
    Serial.println(F("ERR"));
    return;
  }

  uint8_t checksum;
  if (Serial.readBytes((char*)&checksum, 1) != 1) {
    Serial.println(F("ERR"));
    return;
  }

  uint16_t calc = 0;
  for (uint8_t i = 0; i < NN_INPUT_SIZE; i++) calc += pixels[i];
  if ((uint8_t)(calc & 0xFF) != checksum) {
    Serial.println(F("ERR"));
    return;
  }

  int32_t scores[NN_NUM_CLASSES];
  int8_t pred = infer(pixels, scores);

  Serial.print(F("PRED="));
  Serial.print(pred);
  Serial.print(F(" SCORES="));
  for (uint8_t k = 0; k < NN_NUM_CLASSES; k++) {
    Serial.print(scores[k]);
    if (k < NN_NUM_CLASSES - 1) Serial.print(' ');
  }
  Serial.println();
}
