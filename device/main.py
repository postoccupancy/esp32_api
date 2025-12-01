import network, time, ujson
import urequests
from machine import I2C, Pin

from LCD import show_reading

WIFI_SSID = "Knight-MacDonald"
WIFI_PASS = "409Jasper!"
SERVER    = "http://192.168.0.3:8000/ingest"  # your laptop IP
TOKEN     = "change-me"
DEVICE_ID = "esp32-s3-devkit-001"


# ---- Wi-Fi connect ----
def wifi_connect():
    sta = network.WLAN(network.STA_IF)
    sta.active(True)
    if not sta.isconnected():
        sta.connect(WIFI_SSID, WIFI_PASS)
        t0 = time.ticks_ms()
        while not sta.isconnected():
            if time.ticks_diff(time.ticks_ms(), t0) > 15000:
                raise RuntimeError("WiFi timeout")
            time.sleep_ms(200)
    return sta.ifconfig()


# ---- Optional: sync RTC via NTP (if available) ----
def ntp_sync():
    try:
        import ntptime
        ntptime.settime()  # sets RTC to UTC
        print("NTP OK:", time.localtime())
    except Exception as e:
        print("NTP failed:", e)


# ---- AM2320 read (working version) ----
i2c = I2C(0, scl=Pin(9), sda=Pin(10), freq=100000)
ADDR = 0x5C

def am2320_read():
    try:
        i2c.writeto(ADDR, b'')  # wake
    except OSError:
        pass
    time.sleep_ms(10)
    i2c.writeto(ADDR, b'\x03\x00\x04')
    time.sleep_ms(2)
    data = i2c.readfrom(ADDR, 8)
    hum = (data[2] << 8 | data[3]) / 10.0
    t   = (data[4] << 8) | data[5]
    if t & 0x8000:
        t = -(t & 0x7FFF)
    temp_c = t / 10.0
    temp_f = temp_c * 9 / 5 + 32
    return temp_c, temp_f, hum


def iso_ts():
    y, mo, d, hh, mm, ss, _, _ = time.localtime()
    return "%04d-%02d-%02dT%02d:%02d:%02dZ" % (y, mo, d, hh, mm, ss)


# ---- Boot: WiFi + (optional) NTP ----
print("WiFi:", wifi_connect())
ntp_sync()

# LED on GPIO4 as simple digital output
led = Pin(4, Pin.OUT)
led.value(0)  # start off

SENSOR_INTERVAL_S = 2  # seconds

while True:
    try:
        # 1) Read sensor
        temp_c, temp_f, rh = am2320_read()

        # 2) Update LCD with latest reading
        show_reading(temp_c, temp_f, rh)

        # 3) Build payload
        payload = {
            "device_id": DEVICE_ID,
            "ts": iso_ts(),            # may be 2000-* without NTP; server can override
            "temp_c": round(temp_c, 1),
            "temp_f": round(temp_f, 1),
            "rh":     round(rh, 1),
        }

        # 4) POST to FastAPI
        try:
            r = urequests.post(
                SERVER,
                headers={
                    "Content-Type": "application/json",
                    "X-Token": TOKEN,
                },
                data=ujson.dumps(payload),
            )
            status = r.status_code
            r.close()
            print("POST status:", status)
            print("sent:", payload)

            # 5) Blink LED only if POST succeeded (2xx)
            if 200 <= status < 300:
                led.value(1)
                time.sleep_ms(100)  # short visible blink
                led.value(0)

        except Exception as e:
            print("POST failed:", repr(e))
            # no blink if send fails

    except OSError as e:
        print("sensor/WiFi err:", e)
        # brief backoff before trying again
        time.sleep_ms(200)

    # Wait before next reading
    time.sleep(SENSOR_INTERVAL_S)

