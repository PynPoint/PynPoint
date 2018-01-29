import sys

__all__ = ["progress"]

def progress(current, total, message):
    fraction = float(current)/float(total)
    percentage = round(fraction*100., 1)

    sys.stdout.write("%s %s%s \r" % (message, percentage, "%"))
    sys.stdout.flush()

    if percentage == 100.0:
        sys.stdout.write("%s %s \r" % (message, "[DONE]"))
