from sys import platform

# darwin is MacOS
# win32 is windows
# linux or linux2 are linux

def init():
    global os_name
    os_name = platform

def isMac():
    init()
    if os_name == "darwin":
        return True
    else:
        return False

if __name__ == "__main__":
    init()
    print(isMac())