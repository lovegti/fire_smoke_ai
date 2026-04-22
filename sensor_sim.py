import random
import time

history = []

class SensorSim:
    def normal(self):
        return {
            "temp": round(random.uniform(22,28),1),
            "smoke": random.randint(10,30),
            "co": random.randint(5,15),
            "time": time.strftime("%H:%M:%S")
        }

    def fire(self):
        return {
            "temp": round(random.uniform(45,80),1),
            "smoke": random.randint(200,600),
            "co": random.randint(50,120),
            "time": time.strftime("%H:%M:%S")
        }

sensor = SensorSim()

def get_sensor(is_fire):
    d = sensor.fire() if is_fire else sensor.normal()
    global history
    history.append(d)
    history = history[-8:]
    return d, history