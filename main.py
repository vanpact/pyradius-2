#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    .. module:: main
        :platform: Unix, Windows
        :synopsis: This module contains the main function of the application.
        
    .. moduleauthor:: Yves-Rémi Van Eycke <yveycke [at] ulb.ac.be>
"""

import sys
from PyQt4 import QtGui, QtCore
from View import Window

def main():
    """The main function of the program
        
        Returns: 
            int.  The return code::
                0 -- Success!
    """
    app = QtGui.QApplication(sys.argv)
#     styleSheet = QtCore.QFile("darkorange.stylesheet")
#     styleSheet.open(QtCore.QFile.ReadOnly)
#     style = QtCore.QLatin1String(styleSheet.readAll())
#     app.setStyleSheet(style)
    v = Window()
    v.finishedWork.connect(app.quit)
    v.show()
    sys.exit(app.exec_())
    return 0
    
if __name__ == '__main__':
    main()
    