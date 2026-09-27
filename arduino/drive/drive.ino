#include <Servo.h>

// One sketch for the steering motor and the vibration servo.
// The Pi sends one line at 9600 baud:
//   soft left, hard left, soft right, hard right, stop, straight
//
// Center the wheel by hand before powering on. There is no position sensor.
// Each command turns once, holds, then returns once. Repeating a command
// does not turn again, so small timing errors do not walk the center left or right.

#define PWMA 5
#define AIN1 7
#define AIN2 8
#define STBY 9

#define MOTOR_SPEED 80
#define RETURN_SPEED 40

#define TIME_20_DEG  100
#define TIME_45_DEG  150
#define TIME_90_DEG  300

const int SERVO_PIN = 10;
const int CENTER = 90;
const int VIBRATION_AMOUNT = 8;
const int VIBRATION_DELAY = 15;

// 0 = center. Negative = left. Positive = right.
int position = 0;
String pendingCommand = "";
Servo myServo;

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(20);

  pinMode(PWMA, OUTPUT);
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(STBY, OUTPUT);
  digitalWrite(STBY, HIGH);
  stopMotor();

  myServo.attach(SERVO_PIN);
  myServo.write(CENTER);
}

void stopMotor() {
  analogWrite(PWMA, 0);
  digitalWrite(AIN1, HIGH);
  digitalWrite(AIN2, HIGH);
}

void moveLeft(int duration, int speed) {
  digitalWrite(AIN1, LOW);
  digitalWrite(AIN2, HIGH);
  analogWrite(PWMA, speed);
  delay(duration);
  stopMotor();
}

void moveRight(int duration, int speed) {
  digitalWrite(AIN1, HIGH);
  digitalWrite(AIN2, LOW);
  analogWrite(PWMA, speed);
  delay(duration);
  stopMotor();
}

void returnToCenter() {
  if (position < 0) {
    int distance = -position;
    if (distance == 20) {
      moveRight(TIME_20_DEG * 2, RETURN_SPEED);
    } else if (distance == 45) {
      moveRight(TIME_45_DEG * 2, RETURN_SPEED);
    } else if (distance == 90) {
      moveRight(TIME_90_DEG, RETURN_SPEED);
    }
  } else if (position > 0) {
    int distance = position;
    if (distance == 20) {
      moveLeft(TIME_20_DEG * 2, RETURN_SPEED);
    } else if (distance == 45) {
      moveLeft(TIME_45_DEG * 2, RETURN_SPEED);
    } else if (distance == 90) {
      moveLeft(TIME_90_DEG, RETURN_SPEED);
    }
  }
  position = 0;
}

void seek(int target) {
  if (target == position) {
    return;
  }
  returnToCenter();
  if (target == -20) {
    moveLeft(TIME_20_DEG, MOTOR_SPEED);
  } else if (target == -45) {
    moveLeft(TIME_45_DEG, MOTOR_SPEED);
  } else if (target == -90) {
    moveLeft(TIME_90_DEG, MOTOR_SPEED);
  } else if (target == 20) {
    moveRight(TIME_20_DEG, MOTOR_SPEED);
  } else if (target == 45) {
    moveRight(TIME_45_DEG, MOTOR_SPEED);
  }
  position = target;
}

void oscillate(unsigned long duration) {
  unsigned long startTime = millis();
  while (millis() - startTime < duration) {
    myServo.write(CENTER + VIBRATION_AMOUNT);
    delay(VIBRATION_DELAY);
    myServo.write(CENTER - VIBRATION_AMOUNT);
    delay(VIBRATION_DELAY);
  }
  myServo.write(CENTER);
}

void playHaptic(const String& command) {
  if (command == "soft left" || command == "soft right") {
    oscillate(500);
    delay(500);
    oscillate(500);
  } else if (command == "hard left" || command == "hard right") {
    oscillate(1500);
  } else if (command == "stop") {
    oscillate(400);
    delay(200);
    oscillate(400);
    delay(200);
    oscillate(400);
  }
}

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

void perform(const String& command, int target) {
  seek(target);
  do {
    playHaptic(command);
  } while (sameCommand(command));
  returnToCenter();
}

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
    perform(command, -20);
  } else if (command == "hard left") {
    perform(command, -45);
  } else if (command == "soft right") {
    perform(command, 20);
  } else if (command == "hard right") {
    perform(command, 45);
  } else if (command == "stop") {
    perform(command, -90);
  } else if (command == "straight") {
    returnToCenter();
    myServo.write(CENTER);
  }
}
