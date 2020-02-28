import os
import time


def create_unique_logdir(logdir, root_logdir="log/"):
    """
    Creates a unique log directory using the directory name and the time stamp
    Takes in a unqiue directory name and optionally a root directory path
    The root directory path is default to "log/" since all logs should be stored 
    under that directory

    Example:
        > create_unique_logdir("baseline_lstm")
        "log/baseline_lstm_Y2020_M2_D27_h16_m5"
    """
    localtime = time.localtime(time.time())
    time_label = "Y{}_M{}_D{}_h{}_m{}".format(localtime.tm_year, localtime.tm_mon, \
        localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
    return os.path.join(root_logdir, logdir + "_" + time_label)