import gpiozero as gz
import time

pin_led1 = gz.LED(17)
pin_led2 = gz.DigitalOutputDevice(27)

while True:
    pin_led1.on()
    pin_led2.on()
    time.sleep(0.5)

    pin_led1.off()
    pin_led2.off()
    time.sleep(0.5)
