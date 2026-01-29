#include <Servo.h>

// Number of servos
#define NUM_SERVOS 6

// Servo objects
Servo servos[NUM_SERVOS];

// Servo pins (adjust based on your Arduino board)
const int servoPins[NUM_SERVOS] = {3, 5, 6, 9, 10, 11};

// Current servo positions
int currentPositions[NUM_SERVOS] = {90, 90, 90, 90, 90, 90};

// Target servo positions
int targetPositions[NUM_SERVOS] = {90, 90, 90, 90, 90, 90};

// Servo limits (degrees)
const int servoMin[NUM_SERVOS] = {0, 0, 0, 0, 0, 0};
const int servoMax[NUM_SERVOS] = {180, 180, 180, 180, 180, 180};

// Movement speed (degrees per update)
const int moveSpeed = 2;

// Serial communication
const byte START_BYTE = 0xFF;
byte incomingData[8]; // Start + 6 angles + checksum
int dataIndex = 0;
bool receivingData = false;

// LED pin for status
const int LED_PIN = 13;

// Safety
unsigned long lastCommandTime = 0;
const unsigned long COMMAND_TIMEOUT = 1000; // 1 second timeout

void setup() {
  // Initialize serial
  Serial.begin(115200);

  // Initialize LED
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Attach servos
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].attach(servoPins[i]);
    servos[i].write(currentPositions[i]);
  }

  // Startup blink
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(200);
  }

  Serial.println("Arduino Robot Controller Ready");
}

void loop() {
  // Read serial data
  readSerialData();

  // Update servo positions smoothly
  updateServos();

  // Safety check - stop if no commands received
  if (millis() - lastCommandTime > COMMAND_TIMEOUT) {
    // Hold current position (don't move)
    for (int i = 0; i < NUM_SERVOS; i++) {
      targetPositions[i] = currentPositions[i];
    }
    digitalWrite(LED_PIN, LOW);
  } else {
    digitalWrite(LED_PIN, HIGH);
  }

  delay(20); // 50 Hz update rate
}

void readSerialData() {
  while (Serial.available() > 0) {
    byte inByte = Serial.read();

    if (!receivingData) {
      // Look for start byte
      if (inByte == START_BYTE) {
        receivingData = true;
        dataIndex = 0;
        incomingData[dataIndex++] = inByte;
      }
    } else {
      // Collect data
      incomingData[dataIndex++] = inByte;

      // Check if we have complete packet (8 bytes total)
      if (dataIndex >= 8) {
        processPacket();
        receivingData = false;
        dataIndex = 0;
      }
    }
  }
}

void processPacket() {
  // Verify checksum
  int checksum = 0;
  for (int i = 1; i < 7; i++) { // Sum angles only
    checksum += incomingData[i];
  }
  checksum = checksum & 0xFF;

  if (checksum != incomingData[7]) {
    Serial.println("Checksum error");
    return;
  }

  // Extract servo angles
  for (int i = 0; i < NUM_SERVOS; i++) {
    int angle = incomingData[i + 1];

    // Clamp to limits
    angle = constrain(angle, servoMin[i], servoMax[i]);

    targetPositions[i] = angle;
  }

  lastCommandTime = millis();

  // Debug output (optional - comment out for production)
  // Serial.print("Received: ");
  // for (int i = 0; i < NUM_SERVOS; i++) {
  //   Serial.print(targetPositions[i]);
  //   Serial.print(" ");
  // }
  // Serial.println();
}

void updateServos() {
  // Move servos smoothly towards target positions
  for (int i = 0; i < NUM_SERVOS; i++) {
    if (currentPositions[i] < targetPositions[i]) {
      currentPositions[i] =
          min(currentPositions[i] + moveSpeed, targetPositions[i]);
    } else if (currentPositions[i] > targetPositions[i]) {
      currentPositions[i] =
          max(currentPositions[i] - moveSpeed, targetPositions[i]);
    }

    servos[i].write(currentPositions[i]);
  }
}

// Emergency stop function (can be triggered by external button)
void emergencyStop() {
  // Detach all servos
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].detach();
  }

  // Blink LED rapidly
  while (true) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  }
}
