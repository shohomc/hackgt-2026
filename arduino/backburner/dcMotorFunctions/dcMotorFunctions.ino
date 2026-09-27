// ===============================
// TB6612 MOTOR DRIVER
// ===============================

#define PWMA 5
#define AIN1 7
#define AIN2 8
#define STBY 9


// ===============================
// MOTOR SETTINGS
// ===============================

// Adjust these after testing
#define MOTOR_SPEED 80
#define RETURN_SPEED 40

// Approximate time required to move each angle
#define TIME_20_DEG  100
#define TIME_45_DEG  150
#define TIME_90_DEG  300


// ===============================
// SOFTWARE POSITION
// ===============================

// 0 = center
// Negative = left
// Positive = right

int position = 0;

// A command that arrived while the wheel was holding. loop() runs it after the return.
String pendingCommand = "";


// ===============================
// SETUP
// ===============================

void setup() {

  Serial.begin(9600);
  Serial.setTimeout(20);

  pinMode(PWMA, OUTPUT);
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(STBY, OUTPUT);

  digitalWrite(STBY, HIGH);

  // IMPORTANT:
  // Before powering on the Arduino, manually
  // place the wheel in its center position.

  stopMotor();
}


// ===============================
// STOP MOTOR
// ===============================

void stopMotor() {

  analogWrite(PWMA, 0);

  digitalWrite(AIN1, HIGH);
  digitalWrite(AIN2, HIGH);
}


// ===============================
// MOVE LEFT
// ===============================

void moveLeft(int duration, int speed) {

  digitalWrite(AIN1, LOW);
  digitalWrite(AIN2, HIGH);

  analogWrite(PWMA, speed);

  delay(duration);

  stopMotor();
}


// ===============================
// MOVE RIGHT
// ===============================

void moveRight(int duration, int speed) {

  digitalWrite(AIN1, HIGH);
  digitalWrite(AIN2, LOW);

  analogWrite(PWMA, speed);

  delay(duration);

  stopMotor();
}


// ===============================
// SERIAL CHECK
// ===============================

String readLatest() {

  String latest = "";

  while (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      latest = line;
    }
  }

  return latest;
}

// Read the newest line. The same command means do another period.
// A different command is saved for loop().
bool sameCommand(const String& command) {

  String latest = readLatest();

  if (latest == command) {
    return true;
  }

  if (latest.length() > 0) {
    pendingCommand = latest;
  }

  return false;
}


// ===============================
// RETURN TO CENTER
// ===============================

void returnToCenter() {

  if (position < 0) {

    // Currently left of center
    // Move right slowly

    int distance = -position;

    if (distance == 20) {
      moveRight(TIME_20_DEG*2, RETURN_SPEED);
    }
    else if (distance == 45) {
      moveRight(TIME_45_DEG*2, RETURN_SPEED);
    }
    else if (distance == 90) {
      moveRight(TIME_90_DEG, RETURN_SPEED);
    }

  }

  else if (position > 0) {

    // Currently right of center
    // Move left slowly

    int distance = position;

    if (distance == 20) {
      moveLeft(TIME_20_DEG*2, RETURN_SPEED);
    }
    else if (distance == 45) {
      moveLeft(TIME_45_DEG*2, RETURN_SPEED);
    }
    else if (distance == 90) {
      moveLeft(TIME_90_DEG, RETURN_SPEED);
    }
  }

  // We are now assumed to be centered
  position = 0;
}


// ===============================
// SLIGHT LEFT
// ===============================

void slightLeft() {

  do {
    // Start from center, then move 20 degrees left
    returnToCenter();

    moveLeft(TIME_20_DEG, MOTOR_SPEED);

    position = -20;

    delay(3000);
  } while (sameCommand("soft left"));

  returnToCenter();
}


// ===============================
// HARD LEFT
// ===============================

void hardLeft() {

  do {
    returnToCenter();

    moveLeft(TIME_45_DEG, MOTOR_SPEED);

    position = -45;

    delay(3000);
  } while (sameCommand("hard left"));

  returnToCenter();
}


// ===============================
// SLIGHT RIGHT
// ===============================

void slightRight() {

  do {
    returnToCenter();

    moveRight(TIME_20_DEG, MOTOR_SPEED);

    position = 20;

    delay(3000);
  } while (sameCommand("soft right"));

  returnToCenter();
}


// ===============================
// HARD RIGHT
// ===============================

void hardRight() {

  do {
    returnToCenter();

    moveRight(TIME_45_DEG, MOTOR_SPEED);

    position = 45;

    delay(3000);
  } while (sameCommand("hard right"));

  returnToCenter();
}


// ===============================
// BRAKE
// ===============================

void brake() {

  do {
    returnToCenter();

    // Move 90 degrees left
    moveLeft(TIME_90_DEG, MOTOR_SPEED);

    position = -90;

    delay(3000);
  } while (sameCommand("stop"));

  // Since we don't have a sensor, use the
  // 90-degree calibrated time.
  returnToCenter();
}


// ===============================
// TEST PROGRAM
// ===============================

void loop() {

  String command = pendingCommand;
  pendingCommand = "";

  if (command.length() == 0) {
    if (!Serial.available()) {
      return;
    }
    command = Serial.readStringUntil('\n');
    command.trim();
  }

  if (command == "soft left") {
    slightLeft();
  }
  else if (command == "hard left") {
    hardLeft();
  }
  else if (command == "soft right") {
    slightRight();
  }
  else if (command == "hard right") {
    hardRight();
  }
  else if (command == "stop") {
    brake();
  }
  else if (command == "straight") {
    stopMotor();
  }

}