// TB6612 motor driver. Times are approximate degrees and need a road test.

#define PWMA 5
#define AIN1 7
#define AIN2 8
#define STBY 9

#define MOTOR_SPEED 80
#define RETURN_SPEED 40

#define TIME_20_DEG  100
#define TIME_45_DEG  150
#define TIME_90_DEG  300

// 0 = center. Negative = left. Positive = right.
int position = 0;

void setupMotor() {
  pinMode(PWMA, OUTPUT);
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(STBY, OUTPUT);
  digitalWrite(STBY, HIGH);
  stopMotor();
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
    }
  } else if (position > 0) {
    int distance = position;
    if (distance == 20) {
      moveLeft(TIME_20_DEG * 2, RETURN_SPEED);
    } else if (distance == 45) {
      moveLeft(TIME_45_DEG * 2, RETURN_SPEED);
    }
  }
  position = 0;
}

void slightLeft() {
  moveLeft(TIME_20_DEG, MOTOR_SPEED);
  position = -20;
  delay(3000);
  returnToCenter();
}

void hardLeft() {
  moveLeft(TIME_45_DEG, MOTOR_SPEED);
  position = -45;
  delay(3000);
  returnToCenter();
}

void slightRight() {
  moveRight(TIME_20_DEG, MOTOR_SPEED);
  position = 20;
  delay(3000);
  returnToCenter();
}

void hardRight() {
  moveRight(TIME_45_DEG, MOTOR_SPEED);
  position = 45;
  delay(3000);
  returnToCenter();
}

void brake() {
  moveLeft(TIME_90_DEG, MOTOR_SPEED);
  position = -90;
  delay(3000);
  moveRight(TIME_90_DEG, RETURN_SPEED);
  position = 0;
}
