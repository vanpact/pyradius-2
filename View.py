# -*- coding: utf-8 -*-

"""
    .. module:: View
        :platform: Unix, Windows
        :synopsis: module which manage the view of the application.
    .. moduleauthor:: Yves-Rémi Van Eycke <yveycke [at] ulb.ac.be>
"""

from PyQt4 import QtGui, QtCore, QtMultimedia
from VideoWidget import VideoWidget, Movie
import PreTreatments, Treatments #import Applier, CannyTreatment, GaborTreatment
from threading import Thread

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
        
        self.createMenuBar()
        self.createLayout()
#        self.source = movie()
#        QtCore.QObject.connect(self.source, QtCore.SIGNAL('frameChanged'), self.frameChanged)
#        self.source.frameChanged.connect(self.frameChanged)
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
        videoWidget = VideoWidget()
        self.surface = videoWidget.videoSurface()
#        scene = QtGui.QGraphicsScene(self)
#        graphicsView = QtGui.QGraphicsView(scene)
#        self.videoItem = VideoItem(scene)
#        scene.addItem(self.videoItem)
        
        controlLayout = QtGui.QHBoxLayout()
        self.playButton = QtGui.QPushButton('play')
        self.playButton.resize(50, 50)
        self.timeSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.timeSlider)
        self.playButton.clicked.connect(self.toggleProcessVideo)
        self.timeSlider.valueChanged.connect(self.jumpToFrame)
        
#        preTreatmentLayout = QtGui.QHBoxLayout()
        preTreatmentComboBox = QtGui.QComboBox()
        preTreatmentComboBox.addItem('Muscles:Gabor+Sobel')
        preTreatmentComboBox.addItem('Aponeurosis:Sobel+Gabor')
        preTreatmentComboBox.currentIndexChanged.connect(self.chooseTreatment)
        self.chooseTreatment(1)
        controlLayout.addWidget(preTreatmentComboBox)
        
        mainLayout.addWidget(videoWidget)
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
#            self.source.play()
    
    def toggleProcessVideo(self):
        """Pause and play the video processing"""
        self.filterApplied.toggle()
#        self.filterApplied.run()
        if(not(self.filterApplied.wait or self.filterApplied.isRunning())):
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
        self.timeSlider.setValue(self.source.currentPositionRatio()*self.timeSlider.maximum())

    def jumpToFrame(self, value):
        """ Jump to a position in the movie. 
        
        Args: 
            value: A value in the range of the slider.
        Raises:
             AttributeError: if value is to high or negative.
        """
        ratio = float(value)/float(self.timeSlider.maximum())
        if(ratio<1.0 and ratio>0.0):
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
        
    def chooseTreatment(self, index):
        """Add the filters to be applied."""
        self.filterApplied = PreTreatments.Applier(self.source)
        if(index == 0):
#            self.filterApplied = self.filterApplied + PreTreatments.cropTreatment(([90, 330],[50, 565]))
            self.filterApplied = self.filterApplied + PreTreatments.cropTreatment([[140, 270], [50, 565]])
            self.filterApplied = self.filterApplied + PreTreatments.ReduceSizeTreatment()
#            self.filterApplied = self.filterApplied + PreTreatments.LaplacianTreatment()
            self.filterApplied = self.filterApplied + PreTreatments.GaborTreatment()
#            self.filterApplied = self.filterApplied + PreTreatments.CannyTreatment()
            self.filterApplied = self.filterApplied + PreTreatments.SobelTreatment()
            self.filterApplied = self.filterApplied + PreTreatments.changeContrastTreatment(20.0)
            self.filterApplied = self.filterApplied + PreTreatments.ThresholdTreatment(230)
            self.filterApplied = self.filterApplied + PreTreatments.IncreaseSizeTreatment()
            self.filterApplied = self.filterApplied + Treatments.blobDetectionTreatment(offset = [50, 140])
        elif(index == 1):
            self.filterApplied = self.filterApplied + PreTreatments.cropTreatment(([90, 330],[50, 565]))
            self.filterApplied = self.filterApplied + PreTreatments.ReduceSizeTreatment()
#            self.filterApplied = self.filterApplied + PreTreatments.GaborTreatment()
            self.filterApplied = self.filterApplied + PreTreatments.SobelTreatment(dx=0)
            self.filterApplied = self.filterApplied + PreTreatments.changeContrastTreatment(20.0)
            self.filterApplied = self.filterApplied + PreTreatments.ThresholdTreatment(230)
            self.filterApplied = self.filterApplied + PreTreatments.erosionTreatment(size=(14, 1))
            self.filterApplied = self.filterApplied + PreTreatments.IncreaseSizeTreatment()
            self.filterApplied = self.filterApplied + Treatments.AponeurosisDetector(offset = [50, 90])

