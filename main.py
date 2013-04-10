#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    .. module:: main
        :platform: Unix, Windows
        :synopsis: This module contains the main function of the application.
        
    .. moduleauthor:: Yves-Rémi Van Eycke <yveycke [at] ulb.ac.be>
"""

import sys
from PyQt4 import QtGui
from View import Window

def main():
    """The main function of the program
        
        Returns: 
            int.  The return code::
                0 -- Success!
    """
    app = QtGui.QApplication(sys.argv)
    v = Window()
    v.finishedWork.connect(app.quit)
    v.show()
    sys.exit(app.exec_())
    return 0
    
if __name__ == '__main__':
    main()
    