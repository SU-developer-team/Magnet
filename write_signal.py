import usb.core, usb.util, struct

VID, PID = 0x0547, 0x1003          # замените на реальные (Device Manager ▸ Details ▸ Hardware IDs)
dev = usb.core.find(idVendor=VID, idProduct=PID)
dev.set_configuration()

cfg  = dev.get_active_configuration()
intf = cfg[(0, 0)]
usb.util.claim_interface(dev, intf)

ep_out = usb.util.find_descriptor(intf,
    custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) ==
                           usb.util.ENDPOINT_OUT)

ep_in  = usb.util.find_descriptor(intf,
    custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) ==
                           usb.util.ENDPOINT_IN)

# 1️⃣ Стартуем поток, если требуется
START = bytes([0xA5, 0x5A])       # пример команды
if ep_out:                         # не у всех приборов есть OUT-endpoint
    dev.write(ep_out.bEndpointAddress, START, timeout=100)

# 2️⃣ Читаем буфером «с запасом»
PKT = 4096                         # > wMaxPacketSize
while True:
    try:
        data = dev.read(ep_in.bEndpointAddress, PKT, timeout=1000)
        if not data:
            continue               # пустой ответ – ждём дальше
        samples = struct.unpack('<' + 'h'*(len(data)//2), data)
        print(samples[:8], '…')    # отладка
    except usb.core.USBError as e:
        if e.errno in (110, None): # таймаут – не страшно
            print('-- timeout, ждём…')
            continue
        raise                      