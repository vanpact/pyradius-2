# -*- coding: utf-8 -*-

"""
    .. module:: View
        :platform: Unix, Windows
        :synopsis: module which manage the view of the application.
    .. moduleauthor:: Yves-Rémi Van Eycke <yveycke [at] ulb.ac.be>
"""

from PyQt4 import QtGui, QtCore, QtMultimedia
from VideoWidget import VideoWidget, Movie
import PreTreatments, Treatments, numpy #import Applier, CannyTreatment, GaborTreatment

class Window(QtGui.QMainWindow):
    """ This is the class which contains all the GUI."""
    surface = None
    source = Movie()
#    timeSlider = None
    filterApplied = None
    playButton = None
    
    def __init__(self):
        """ Create the view object."""
        super(Window, self).__init__()
        self.initUI()

    def initUI(self):
        """ Initialize the main windows of the application. """
        self.resize(640, 480)
        self.centerWindow
        self.setWindowTitle('Line')
        self.setWindowIcon(QtGui.QIcon('Images/Icon.png')) 
        QtCore.QTextCodec.setCodecForCStrings(QtCore.QTextCodec.codecForName('UTF-8'))
        
        self.createMenuBar()
        self.createLayout()
#        self.source = movie()
#        QtCore.QObject.connect(self.source, QtCore.SIGNAL('frameChanged'), self.frameChanged)
#        self.source.frameChanged.connect(self.frameChanged)
        self.filterApplied = PreTreatments.Applier.getInstance()
        self.filterApplied.frameComputed.connect(self.frameChanged)
        self.statusBar().showMessage('Ready')
        
    def centerWindow(self):
        """ Center the windows on the screen of the user. """
        windowsGeometry = self.frameGeometry()
        ScreenCenter = QtGui.QDesktopWidget().availableGeometry().center()
        windowsGeometry.moveCenter(ScreenCenter)
        self.move(windowsGeometry.topLeft())       
     
    def createMenuBar(self):
        """ Create the Menu bar with its menu and add the actions to the menus. """
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(self.createAction('&Exit', QtGui.qApp.quit, 'Ctrl+Q', 'Exit application'))
        fileMenu.addAction(self.createAction('&Open', self.openFile, 'Ctrl+O', 'Open a movie'))
        fileMenu.addAction(self.createAction('&Save Frame', self.captureImage, 'Ctrl+P', 'Save the current frame in the movie'))
        
    def createAction(self, name, funct, shortcut='', statusTip=''):
        """Create the actions used, for example, in the menu bar and return it.
        
        Args: 
            name: The name of the action.
            funct: the function to be triggered by the action.
        
        Kwargs:
            shortcut: the shortcut for the action.
            statusTip: a tip which will appear when hovering the action.
        
        Returns: 
            The QAction object
        
         Raises:
             AttributeError: if there is the name or the function send is none.
        """
        
        if name != None:
            action = QtGui.QAction(name, self)
        else:
            raise AttributeError('The action need a name!')
            
        if shortcut != None:
            action.setShortcut(shortcut)
        
        if statusTip != None:
            action.setStatusTip(statusTip)
        
        if funct != None :
            action.triggered.connect(funct)
        else:
            raise AttributeError('The action need a callable or a signal for when triggered!')
        
        return action

    def createLayout(self):
        """ Create the layout of the window. """
        centralWidget = QtGui.QWidget(self)
        mainLayout = QtGui.QVBoxLayout(centralWidget)
        self.videoWidget = VideoWidget()
        self.surface = self.videoWidget.videoSurface()
#        scene = QtGui.QGraphicsScene(self)
#        graphicsView = QtGui.QGraphicsView(scene)
#        self.videoItem = VideoItem(scene)
#        scene.addItem(self.videoItem)
        
        controlLayout = QtGui.QHBoxLayout()
        self.playButton = QtGui.QPushButton('play')
        self.playButton.resize(50, 50)
#        self.timeSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.progressBar = QtGui.QProgressBar()
        self.progressBar.setOrientation(QtCore.Qt.Horizontal)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.progressBar)
        self.playButton.clicked.connect(self.toggleProcessVideo)
        self.progressBar.valueChanged.connect(self.jumpToFrame)
        
#        preTreatmentLayout = QtGui.QHBoxLayout()
        self.preTreatmentComboBox = QtGui.QComboBox()
        self.preTreatmentComboBox.addItem('Muscles:Gabor+Sobel')
        self.preTreatmentComboBox.addItem('Aponeurosis:Sobel+Gabor')
#        self.preTreatmentComboBox.currentIndexChanged[int].connect(self.chooseTreatment)
#        self.chooseTreatment(0)
        controlLayout.addWidget(self.preTreatmentComboBox)
        
        mainLayout.addWidget(self.videoWidget)
#        mainLayout.addWidget(graphicsView)
        mainLayout.insertLayout(1, controlLayout)
#        mainLayout.insertLayout(2, preTreatmentLayout)
        self.setCentralWidget(centralWidget)
        
    def openFile(self):
        """ Show a dialog where there user can pick a file. """
        fileName = QtCore.QString(QtGui.QFileDialog.getOpenFileName(self, "Open Movie", QtCore.QDir.homePath()))
        if (not (fileName.isEmpty())):
            self.surface.stop()
#            self.videoItem.stop()
            self.source.setMovie(str(fileName))
            self.filterApplied.setSource(self.source)
            self.firstTimePlayed = True;
#            self.source.play()
    
    def toggleProcessVideo(self):
        """Pause and play the video processing"""
        if(self.filterApplied.wait):
            self.playButton.setText("Play")
        if(self.firstTimePlayed):
            self.firstTimePlayed = False
            self.filterApplied.applyOne()
            self.launchTutorial()
        else:
            self.filterApplied.toggle()
            self.filterApplied.run()
            if(not(self.filterApplied.wait or self.filterApplied.isRunning())):
                    self.preTreatmentComboBox.setEnabled(False)
                    self.playButton.setText("Pause")
                    self.filterApplied.start(QtCore.QThread.HighestPriority)
            
        
    def frameChanged(self):
        """ Update the view when the frame has changed. """
#        frame = self.source.currentFrame()
        frame = self.filterApplied.getLastComputedFrame()
        if (not(frame.isValid())):
            QtGui.QMessageBox(QtGui.QMessageBox.Critical, 'Error', 'Frame not valid!', QtGui.QMessageBox.Ok, self).show()
            return False;
        currentFormat = self.surface.surfaceFormat()
#        currentFormat = self.videoItem.surfaceFormat()
        if (frame.pixelFormat() != currentFormat.pixelFormat() or frame.size() != currentFormat.frameSize()):
            fmt = QtMultimedia.QVideoSurfaceFormat(frame.size(), frame.pixelFormat())
            if (not(self.surface.start(fmt))):
#            if (not(self.videoItem.start(fmt))):
                QtGui.QMessageBox(QtGui.QMessageBox.Critical, 'Error', 'Surface could not start!', QtGui.QMessageBox.Ok, self).show()
                return False
        if (not(self.surface.present(frame))):
#        if (not(self.videoItem.present(frame))):
            self.surface.stop()
#            self.videoItem.stop()
        self.progressBar.setValue(self.source.currentPositionRatio()*self.progressBar.maximum())

    def jumpToFrame(self, value):
        """ Jump to a position in the movie. 
        
        Args: 
            value: A value in the range of the slider.
        Raises:
             AttributeError: if value is to high or negative.
        """
        ratio = float(value)/float(self.progressBar.maximum())
        if(ratio<=1.0 and ratio>=0.0):
            self.source.jumpToFrame(ratio)
        else:
            raise ValueError("Trying to jump at a position which does not exists")
         
    def captureImage(self):
        """ Take a snapshot of the video and save it. """
#        snapshot = self.source.currentImage()
        snapshot = self.filterApplied.getLastComputedFrame()
        fileName = QtGui.QFileDialog.getSaveFileName(self, "save an image", QtCore.QDir.homePath())
        snapshot.save(fileName)
        self.source.play()
        
    def getsurfacePosition(self):
        p=QtGui.QCursor.pos()
        p=self.videoWidget.mapFromGlobal(p)
        if(self.savedPoint == None):
            self.savedPoint = p
        else:
            self.videoWidget.appendLine(self.savedPoint, p)
            self.savedPoint = None
        
        if(self.videoWidget.getLineNumbers() >=2):
            self.videoWidget.clicked.disconnect(self.getsurfacePosition)
            self.chooseTreatment(self.preTreatmentComboBox.currentIndex())
            self.toggleProcessVideo()
            
    def launchTutorial(self):
        if(self.preTreatmentComboBox.currentIndex() == 0):
            self.tutorial1()
        if(self.preTreatmentComboBox.currentIndex() == 1):
            self.tutorial1()
            
    def tutorial1(self):
        QtGui.QMessageBox(QtGui.QMessageBox.Information, 'Information nécessaire', 
                          'La méthode choisie nécessite des informations de la part de l\'utilisateur.', 
                          QtGui.QMessageBox.Ok, self).show()
        self.savedPoint = None
        self.videoWidget.clicked.connect(self.getsurfacePosition)
        
        
    def chooseTreatment(self, index):
        """Add the filters to be applied."""
        if(index == 0):
            self.filterApplied = PreTreatments.Applier.getInstance(self.source, 3)
#            self.filterApplied = self.filterApplied.add(PreTreatments.cropTreatment(([90, 330],[50, 565])), 0)
            l = self.videoWidget.getLines()
            oldminx = 100000
            oldmaxx = 0
            oldminy = 100000
            oldmaxy = 0
            for line in l:
                newminx = min(line[0].x(), line[1].x())
                oldminx = min(newminx, oldminx)
                newmaxx = max(line[0].x(), line[1].x())
                oldmaxx = max(newmaxx, oldmaxx)
                newminy = min(line[0].y(), line[1].y())
                oldminy = min(newminy, oldminy)
                newmaxy = max(line[0].y(), line[1].y())
                oldmaxy = max(newmaxy, oldmaxy)
            sortedLines = []
            for line in l:
                sortedLines.append((QtCore.QPoint(min(line[0].x(), line[1].x()), min(line[0].y(), line[1].y())),QtCore.QPoint(max(line[0].x(), line[1].x()), max(line[0].y(), line[1].y()))))
            self.filterApplied = self.filterApplied.add(PreTreatments.cropTreatment(([oldminx, oldminy], [oldmaxx, oldmaxy])), 0)#[[140, 270], [50, 565]]), 0)
#            self.filterApplied = self.filterApplied.add(PreTreatments.ReduceSizeTreatment(), 0)
#            self.filterApplied = self.filterApplied.add(PreTreatments.LaplacianTreatment(), 0)
            self.filterApplied = self.filterApplied.add(PreTreatments.GaborTreatment(ksize = 31, sigma = 1.5, lambd = 15, gamma = 0.02, psi = 0), 0)
#            self.filterApplied = self.filterApplied.add(PreTreatments.CannyTreatment(), 0)
            self.filterApplied = self.filterApplied.add(PreTreatments.SobelTreatment(dx=1, dy=1, kernelSize=7, scale=1, delta=0), 0)
##            self.filterApplied = self.filterApplied .add(PreTreatments.changeContrastTreatment(7.0), 0)
            self.filterApplied = self.filterApplied.add(PreTreatments.ThresholdTreatment(-1), 0)
#            self.filterApplied = self.filterApplied.add(PreTreatments.dilationTreatment(size=(2, 2)), 0)
##            self.filterApplied = self.filterApplied.add(PreTreatments.CannyTreatment(), 0)
#            self.filterApplied = self.filterApplied.add(PreTreatments.IncreaseSizeTreatment(), 0)
            self.filterApplied = self.filterApplied.add(Treatments.blobDetectionTreatment(offset = [sortedLines[0][0].x(), sortedLines[0][0].y()]), 0)
            
#            self.filterApplied = self.filterApplied.add(PreTreatments.cropTreatment(([sortedLines[0][0].x(), sortedLines[0][0].y()-30], [sortedLines[0][1].x(), sortedLines[0][1].y()+30])), 1)
##            self.filterApplied = self.filterApplied.add(PreTreatments.ReduceSizeTreatment(), 1)
#            self.filterApplied = self.filterApplied.add(PreTreatments.GaborTreatment(ksize = 31, sigma = 1.5, lambd = 15, gamma = 0.02, psi = 0), 1)
##            self.filterApplied = self.filterApplied.add(PreTreatments.CannyTreatment(), 1)
##            self.filterApplied = self.filterApplied.add(PreTreatments.IncreaseSizeTreatment(), 1)
##            self.filterApplied = self.filterApplied.add(Treatments.aponeurosisHough(offset = [sortedLines[0][0].x(), sortedLines[0][0].y()-30], angle = numpy.degrees(numpy.arctan(numpy.float(l[0][1].y()-l[0][0].y())/numpy.float(l[0][1].x()-l[0][0].x())))), 1)
#            
#            self.filterApplied = self.filterApplied.add(PreTreatments.cropTreatment(([sortedLines[1][0].x(), sortedLines[1][0].y()-30], [sortedLines[1][1].x(), sortedLines[1][1].y()+30])), 2)
##            self.filterApplied = self.filterApplied.add(PreTreatments.rotationTreatment(angle = numpy.degrees(numpy.arctan(numpy.float(l[1][1].y()-l[1][0].y())/numpy.float(l[1][1].x()-l[1][0].x())))), 2)
##            self.filterApplied = self.filterApplied.add(PreTreatments.ReduceSizeTreatment(), 2)
#            a = numpy.arctan(numpy.float(l[1][1].y()-l[1][0].y())/numpy.float(l[1][1].x()-l[1][0].x()))
#            self.filterApplied = self.filterApplied.add(PreTreatments.GaborTreatment(ksize = 31, sigma = 1.5, lambd = 7, gamma = 0.02, psi = 0, angleToProcess=[a]), 2)
##            self.filterApplied = self.filterApplied.add(PreTreatments.CannyTreatment(), 2)
##            self.filterApplied = self.filterApplied.add(PreTreatments.IncreaseSizeTreatment(), 2)
##            self.filterApplied = self.filterApplied.add(PreTreatments.ThresholdTreatment(100), 2)
#            self.filterApplied = self.filterApplied.add(Treatments.aponeurosisHough(offset = [l[1][0].x(), l[1][0].y()-30], angle = a), 2)

#            self.filterApplied = self.filterApplied.add(PreTreatments.cropTreatment(([90, 330],[50, 565])), 1)
            self.filterApplied = self.filterApplied.add(PreTreatments.cropTreatment(([sortedLines[1][0].x(), sortedLines[1][0].y()-30], [sortedLines[1][1].x(), sortedLines[1][1].y()+30])), 1)
#            self.filterApplied = self.filterApplied.add(PreTreatments.cropTreatment(([oldminx, oldminy-20], [oldmaxx, oldmaxy+20])), 1)
            self.filterApplied = self.filterApplied.add(PreTreatments.ReduceSizeTreatment(), 1)
            self.filterApplied = self.filterApplied.add(PreTreatments.GaborTreatment(ksize = 31, sigma = 1.5, lambd = 15, gamma = 0.02, psi = 0), 1)
            self.filterApplied = self.filterApplied.add(PreTreatments.SobelTreatment(dx=0, dy=1, kernelSize=7, scale=1, delta=0), 1)
#            self.filterApplied = self.filterApplied.add(PreTreatments.changeContrastTreatment(20.0), 1)
            self.filterApplied = self.filterApplied.add(PreTreatments.ThresholdTreatment(-1), 1)
#            self.filterApplied = self.filterApplied.add(PreTreatments.erosionTreatment(size=(35, 1)), 1)
#            self.filterApplied = self.filterApplied.add(PreTreatments.erosionTreatment(size=(14, 1)), 1)
#            self.filterApplied = self.filterApplied.add(PreTreatments.erosionTreatment(size=(7, 1)), 1)
            self.filterApplied = self.filterApplied.add(PreTreatments.IncreaseSizeTreatment(), 1)
            a = numpy.arctan(numpy.float(l[1][1].y()-l[1][0].y())/numpy.float(l[1][1].x()-l[1][0].x()))
            self.filterApplied = self.filterApplied.add(Treatments.AponeurosisDetector(offset = [l[0][0].x(), l[0][0].y()-20], angle=a), 1)#[50, 90]), 1)
        if(index == 1):
            self.filterApplied = PreTreatments.Applier.getInstance(self.source, 1)
            l = self.videoWidget.getLines()
            oldminx = 100000
            oldmaxx = 0
            oldminy = 100000
            oldmaxy = 0
            for line in l:
                newminx = min(line[0].x(), line[1].x())
                oldminx = min(newminx, oldminx)
                newmaxx = max(line[0].x(), line[1].x())
                oldmaxx = max(newmaxx, oldmaxx)
                newminy = min(line[0].y(), line[1].y())
                oldminy = min(newminy, oldminy)
                newmaxy = max(line[0].y(), line[1].y())
                oldmaxy = max(newmaxy, oldmaxy)
            self.filterApplied = self.filterApplied.add(PreTreatments.cropTreatment([[140, 270], [50, 565]]), 0)
#            self.filterApplied = self.filterApplied.add(PreTreatments.cropTreatment([[140, 350], [50, 565]]), 0)
            self.filterApplied = self.filterApplied.add(PreTreatments.GaborTreatment(ksize = 31, sigma = 1.5, lambd = 15, gamma = 0.02, psi = 0), 0)
#            self.filterApplied = self.filterApplied.add(PreTreatments.LaplacianTreatment(), 0)
#            self.filterApplied = self.filterApplied.add(PreTreatments.DOHTreatment(), 0)
#            self.filterApplied = self.filterApplied.add(PreTreatments.CannyTreatment(L2gradient=False), 0)
            self.filterApplied = self.filterApplied.add(PreTreatments.SobelTreatment(dx=1, dy=1, kernelSize=7, scale=1, delta=0), 0)
            self.filterApplied = self.filterApplied.add(PreTreatments.ThresholdTreatment(-1), 0)
#            self.filterApplied = self.filterApplied.add(PreTreatments.DilationTreatment(size=(2, 2)), 0)
            self.filterApplied = self.filterApplied.add(Treatments.testRadon(offset = [50, 140]))
            