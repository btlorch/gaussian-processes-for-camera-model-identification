import getpass
import os


FULL_RESOLUTION = "full_resolution"
LIBSVM_DIR = "libsvm_dir"

# Detector names
GAUSSIAN_PROCESS_CLASSIFIER = "gaussian_process_classifier"
SECURE_SVM = "secure_svm"
PI_SVM = "pi_svm"


constants = dict()


nodename = os.uname().nodename
username = getpass.getuser()
if "leifur" == nodename:
    constants[LIBSVM_DIR] = os.path.expanduser("~/i1/local/libsvm-openset")
elif "faui1pc54" == nodename:
    constants[LIBSVM_DIR] = os.path.expanduser("~/local/libsvm-openset")
