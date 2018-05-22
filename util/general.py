
import logging
from logging.handlers import RotatingFileHandler
import subprocess
import os


def get_logger(name: str, debug_log_file_name: str): # -> logging.Logger:
    alogger = logging.getLogger(name)
    alogger.setLevel(logging.DEBUG) # CAREFUL ==> need this, otherwise everybody chokes!
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s [%(module)s.%(funcName)s:%(lineno)d => %(message)s]')
    #
    create_debug_handler = False
    # fh = logging.FileHandler(debug_log_file_name)
    fh = RotatingFileHandler(debug_log_file_name, mode='a', maxBytes=5 * 1024 * 1024, backupCount=2, encoding=None, delay=0)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    if not len(alogger.handlers):
        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        # and add it to logger
        alogger.addHandler(ch)

        # we need a debug handler: let's flag our needs:
        create_debug_handler = True

        print("Logger created")
    else:
        print("Logger retrieved")
        # did the file handler change names?
        curr_debug_handler = alogger.handlers[1]
        if curr_debug_handler.baseFilename != fh.baseFilename:
            print("Changing log file names; was '{}', switching to '{}'".format(curr_debug_handler.baseFilename,
                                                                                fh.baseFilename))
            alogger.removeHandler(curr_debug_handler)
            # we need a debug handler: let's flag our needs:
            create_debug_handler = True
        else:
            # the debug handler we have is all good!
            create_debug_handler = False

    # If we need a debug handler, let's create it!
    if create_debug_handler:
        print("Creating debug handler at '{}'".format(fh.baseFilename))
        alogger.addHandler(fh)

    s = "'{}': logging 'INFO'+ logs to Console, 'DEBUG'+ logs to '{}'".format(alogger.name, alogger.handlers[1].baseFilename)
    print(s)
    alogger.info(s)
    alogger.debug(s)
    return alogger


def get_git_root() -> str:
    """
    Gets git root of a project. If the code is not on git, returns empty string.
    See http://stackoverflow.com/questions/22081209/find-the-root-of-the-git-repository-where-the-file-lives
    """
    try:
        return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip()\
            .decode("utf-8")  # conversion to string
    except:
        return ""


def append_to_git_root(what: str, alternate_root: str) -> str:
    """
    Appends a path to git root, or to an alternate path (if this code is not running
    on a git-controlled environment)
    :param what: a path
    :param alternate_root: a directory where to append if git root is not defined
    :return: a path
    """
    git_root = get_git_root()
    if (git_root == ''):
        return os.path.join(alternate_root, what)
    else:
        return os.path.join(git_root, what)


def get_free_file_name(a_dir: str, root: str, ext: str) -> str:
    """Finds a file name that hasn't been used."""
    fcounter = 1
    fname = os.path.join(a_dir, "%s_%d.%s" % (root, fcounter, ext))
    while os.path.isfile(fname):
        fcounter += 1
        fname = os.path.join(a_dir, "%s_%d.%s" % (root, fcounter, ext))
    return fname

