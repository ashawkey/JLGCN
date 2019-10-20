import signal
import os
import time

def test_DelayedKeyboardInterrupt():
    from utils import DelayedKeyboardInterrupt
    print("Start! (Please Press Ctrl+C to test)")
    with DelayedKeyboardInterrupt():
        for i in range(5):
            print(i)
    print("Finished!")


def test_send_mail():
    from utils import EmailSender
    print("Send email test")
    sender = EmailSender(username="xxxxxx@126.com", password="xxxxxx")
    sender.send("xxxxxx@126.com", ["xxxxxx@xxx.xx"], subject="report", message="test", files=[])
    print("Email sent!")


if __name__ == "__main__":
    test_send_mail()

