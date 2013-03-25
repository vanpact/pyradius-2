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
        self.OutputFile = None
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
        self.videoWidget.setStringStartPosition((self.videoWidget.size().width()*0.9, self.videoWidget.size().height()))
#        scene = QtGui.QGraphicsScene(self)
#        graphicsView = QtGui.QGraphicsView(scene)
#        self.videoItem = VideoItem(scene)
#        scene.addItem(self.videoItem)
        
        self.controlLayout = QtGui.QVBoxLayout()
        self.basicControlLayout = QtGui.QHBoxLayout()
        self.controlLayout.addLayout(self.basicControlLayout)
        self.playButton = QtGui.QPushButton('play')
        self.playButton.resize(50, 50)
#        self.timeSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.progressBar = QtGui.QProgressBar()
        self.progressBar.setOrientation(QtCore.Qt.Horizontal)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.basicControlLayout.addWidget(self.playButton)
        self.basicControlLayout.addWidget(self.progressBar)
        self.playButton.clicked.connect(self.toggleProcessVideo)
        self.progressBar.valueChanged.connect(self.jumpToFrame)
        
#        preTreatmentLayout = QtGui.QHBoxLayout()
        self.treatmentComboBox = QtGui.QComboBox()
        self.treatmentComboBox.addItem('Fit ellipsoïde')
        self.treatmentComboBox.addItem('Radon transform')
        self.treatmentComboBox.addItem('Lucas-Kanade')
        self.treatmentComboBox.addItem('Seam Carving')
        self.outputLabel = QtGui.QLabel('Sauver les résultats :')
        self.outputCheckBox = QtGui.QCheckBox()
        self.methodOptionLayout= QtGui.QHBoxLayout()
        self.controlLayout.addLayout(self.methodOptionLayout)
        self.treatmentComboBox.currentIndexChanged.connect(self.loadCorrectInterface)
        self.outputCheckBox.stateChanged.connect(self.chooseOutputFile)
        self.createEllipsoidMethodInterface()
        self.createRadonMethodInterface()
        self.createLKMethodInterface()
        self.createSCMethodInterface()
#        self.treatmentComboBox.currentIndexChanged[int].connect(self.chooseTreatment)
#        self.chooseTreatment(0)
        self.basicControlLayout.addWidget(self.treatmentComboBox)
        self.basicControlLayout.addWidget(self.outputLabel)
        self.basicControlLayout.addWidget(self.outputCheckBox)
        
        mainLayout.addWidget(self.videoWidget)
#        mainLayout.addWidget(graphicsView)
        mainLayout.insertLayout(1, self.controlLayout)
#        mainLayout.insertLayout(2, preTreatmentLayout)
        self.setCentralWidget(centralWidget)
        self.treatmentComboBox.currentIndexChanged.emit(0)
    
    def createEllipsoidMethodInterface(self):
        self.ellipsoidOptionWidget = QtGui.QGroupBox("Contrôles spécifiques à la méthode:")
        self.ellipsoidOptionWidget.hide()
        self.controlLayout.addWidget(self.ellipsoidOptionWidget)
        self.ellipsoidOptionLayout = QtGui.QHBoxLayout()
        self.ellipsoidOptionWidget.setLayout(self.ellipsoidOptionLayout)
        self.ellipsoidSkipFrameLabel = QtGui.QLabel("Nombre d'images à passer:")
        self.ellipsoidSkipFrameLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.ellipsoidSkipFrameSpinBox = QtGui.QSpinBox()
        self.ellipsoidSkipFrameSpinBox.setMinimum(0)
        self.ellipsoidSkipFrameSpinBox.setMaximum(5)
        self.ellipsoidOptionLayout.addWidget(self.ellipsoidSkipFrameLabel)
        self.ellipsoidOptionLayout.addWidget(self.ellipsoidSkipFrameSpinBox)
        self.ellipsoidOptionLayout.addStretch()

        
    def createRadonMethodInterface(self):
        self.radonOptionWidget = QtGui.QGroupBox("Contrôles spécifiques à la méthode:")
        self.radonOptionWidget.hide()
        self.controlLayout.addWidget(self.radonOptionWidget)
        self.radonOptionLayout = QtGui.QHBoxLayout()
        self.radonOptionWidget.setLayout(self.radonOptionLayout)
        self.multipleRadonLabel = QtGui.QLabel("Prendre plus d'échantillons:")
        self.multipleRadonLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.multipleRadonCheckBox = QtGui.QCheckBox()
        self.radonSkipFrameLabel = QtGui.QLabel("Nombre d'images à passer:")
        self.radonSkipFrameLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.radonSkipFrameSpinBox = QtGui.QSpinBox()
        self.radonSkipFrameSpinBox.setMinimum(0)
        self.radonSkipFrameSpinBox.setMaximum(24)
        self.radonOptionLayout.addWidget(self.multipleRadonLabel)
        self.radonOptionLayout.addWidget(self.multipleRadonCheckBox)
        self.radonOptionLayout.addWidget(self.radonSkipFrameLabel)
        self.radonOptionLayout.addWidget(self.radonSkipFrameSpinBox)
        self.radonOptionLayout.addStretch()
        
    def createLKMethodInterface(self):
        self.LKOptionWidget = QtGui.QGroupBox("Contrôles spécifiques à la méthode:")
        self.LKOptionWidget.hide()
        self.controlLayout.addWidget(self.LKOptionWidget)
        self.LKOptionLayout = QtGui.QHBoxLayout()
        self.LKOptionWidget.setLayout(self.LKOptionLayout)
        self.LKSkipFrameLabel = QtGui.QLabel("Nombre d'images à passer:")
        self.LKSkipFrameLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.LKSkipFrameSpinBox = QtGui.QSpinBox()
        self.LKSkipFrameSpinBox.setMinimum(0)
        self.LKSkipFrameSpinBox.setMaximum(5)
        self.LKOptionLayout.addWidget(self.LKSkipFrameLabel)
        self.LKOptionLayout.addWidget(self.LKSkipFrameSpinBox)
        self.LKOptionLayout.addStretch()
    
    def createSCMethodInterface(self):
        self.SCOptionWidget = QtGui.QGroupBox("Contrôles spécifiques à la méthode:")
        self.SCOptionWidget.hide()
        self.controlLayout.addWidget(self.SCOptionWidget)
        self.SCOptionLayout = QtGui.QHBoxLayout()
        self.SCOptionWidget.setLayout(self.SCOptionLayout)
        self.SCSkipFrameLabel = QtGui.QLabel("Nombre d'images à passer:")
        self.SCSkipFrameLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.SCSkipFrameSpinBox = QtGui.QSpinBox()
        self.SCSkipFrameSpinBox.setMinimum(0)
        self.SCSkipFrameSpinBox.setMaximum(5)
        self.SCOptionLayout.addWidget(self.SCSkipFrameLabel)
        self.SCOptionLayout.addWidget(self.SCSkipFrameSpinBox)
        self.SCOptionLayout.addStretch()
        
    def loadCorrectInterface(self):
        self.hideAllInterface()
        if(self.treatmentComboBox.currentIndex()==0):
            self.loadEllipsoidTransformInterface()
        if(self.treatmentComboBox.currentIndex()==1):
            self.loadRadonTransformInterface()
        if(self.treatmentComboBox.currentIndex()==2):
            self.loadLKTransformInterface()
        if(self.treatmentComboBox.currentIndex()==3):
            self.loadSCTransformInterface()
       
    def hideAllInterface(self):
        for i in range(self.controlLayout.count()):
            possibleWidget = self.controlLayout.itemAt(i)
            if(possibleWidget.widget()):
                possibleWidget.widget().hide()
    
    def loadEllipsoidTransformInterface(self):
        self.ellipsoidOptionWidget.show()
        
    def loadRadonTransformInterface(self):
        self.radonOptionWidget.show()
    
    def loadLKTransformInterface(self):
        self.LKOptionWidget.show()
    
    def loadSCTransformInterface(self):
        self.SCOptionWidget.show()
        
    def chooseOutputFile(self):  
        if(self.outputCheckBox.isChecked()):
            fileName = QtCore.QString(QtGui.QFileDialog.getSaveFileName(self, "Ouvrir le fichier de sortie", QtCore.QDir.homePath(), 'Text files (*.txt)'))
            if (not (fileName.isEmpty())):
                self.OutputFile = open(str(fileName), 'w', 0)
             
    def openFile(self):
        """ Show a dialog where there user can pick a file. """
        fileName = QtCore.QString(QtGui.QFileDialog.getOpenFileName(self, "ouvrir la vidéo", QtCore.QDir.homePath()))
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
                    self.treatmentComboBox.setEnabled(False)
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
        angle = self.filterApplied.getLastComputedAngle()
        self.videoWidget.setString(str(round(angle, 2))+ "°")
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
        if(self.videoWidget.getPointNumbers() == 0):
            self.videoWidget.appendPoint(p)
        else:
            self.videoWidget.appendLine(self.videoWidget.popLastPoint(), p)
        
        if(self.videoWidget.getLineNumbers() >=self.lineNumber):
            self.videoWidget.clicked.disconnect(self.getsurfacePosition)
            self.chooseTreatment(self.treatmentComboBox.currentIndex())
            self.videoWidget.PointsToDraw = False
            self.videoWidget.LinesToDraw = False
            self.toggleProcessVideo()
            
    def launchTutorial(self):
        if(self.treatmentComboBox.currentIndex() == 0):
            self.tutorial1()
        if(self.treatmentComboBox.currentIndex() == 1):
            self.tutorial1()
        if(self.treatmentComboBox.currentIndex() == 2):
            self.tutorialmuscle()
        if(self.treatmentComboBox.currentIndex() == 3):
            self.chooseTreatment(self.treatmentComboBox.currentIndex())
            self.toggleProcessVideo()
            
    def tutorial1(self):
        QtGui.QMessageBox(QtGui.QMessageBox.Information, 'Information nécessaire', 
                          'La méthode choisie nécessite 2 informations de la part de l\'utilisateur.', 
                          QtGui.QMessageBox.Ok, self).show()
        self.lineNumber = 2
        self.videoWidget.clicked.connect(self.getsurfacePosition)
        
    def tutorialmuscle(self):
        QtGui.QMessageBox(QtGui.QMessageBox.Information, 'Information nécessaire', 
                          'La méthode choisie nécessite 3 informations de la part de l\'utilisateur.', 
                          QtGui.QMessageBox.Ok, self).show()
        self.lineNumber = 3
        self.videoWidget.clicked.connect(self.getsurfacePosition)
                
    def chooseTreatment(self, index):
        """Add the filters to be applied."""
        if(index == 0):
            self.filterApplied = PreTreatments.Applier.getInstance(self.source, nrChannel=3, nrSkipFrame = self.ellipsoidSkipFrameSpinBox.value(), computType = PreTreatments.computationType.pennationAngleComputation)
            l = self.videoWidget.getLines()
            self.filterApplied.setWriteResults(self.outputCheckBox.isChecked(), self.OutputFile)
            self.filterApplied = self.filterApplied.add(Treatments.blobDetectionTreatment(lines=l), 0)
            self.filterApplied = self.filterApplied.add(Treatments.AponeurosisTracker(lines=l[0]), 1)
            self.filterApplied = self.filterApplied.add(Treatments.AponeurosisTracker(lines=l[1]), 2)
        if(index == 1):
            self.filterApplied = PreTreatments.Applier.getInstance(self.source, nrChannel=3, nrSkipFrame = self.radonSkipFrameSpinBox.value(), computType = PreTreatments.computationType.pennationAngleComputation)
            l = self.videoWidget.getLines()
            self.filterApplied.setWriteResults(self.outputCheckBox.isChecked(), self.OutputFile)
            self.filterApplied = self.filterApplied.add(Treatments.testRadon(lines=l, manyCircles=self.multipleRadonCheckBox), 0)
            self.filterApplied = self.filterApplied.add(Treatments.AponeurosisTracker(lines=l[0]), 1)
            self.filterApplied = self.filterApplied.add(Treatments.AponeurosisTracker(lines=l[1]), 2)
        if(index == 2):
            self.filterApplied = PreTreatments.Applier.getInstance(self.source, nrChannel=3, nrSkipFrame = self.radonSkipFrameSpinBox.value(), computType = PreTreatments.computationType.pennationAngleComputation)
            l = self.videoWidget.getLines()
            self.filterApplied.setWriteResults(self.outputCheckBox.isChecked(), self.OutputFile)
            self.filterApplied = self.filterApplied.add(Treatments.MuscleTracker(lines=l[0:2], fiber=l[2]), 0)
            self.filterApplied = self.filterApplied.add(Treatments.AponeurosisTracker(lines=l[0]), 1)
            self.filterApplied = self.filterApplied.add(Treatments.AponeurosisTracker(lines=l[1]), 2)
        if(index == 3):
            self.filterApplied = PreTreatments.Applier.getInstance(self.source, nrChannel=1, nrSkipFrame = self.ellipsoidSkipFrameSpinBox.value(), computType = PreTreatments.computationType.JunctionComputation)
            self.filterApplied.setWriteResults(self.outputCheckBox.isChecked(), self.OutputFile)
            self.filterApplied = self.filterApplied.add(Treatments.seamCarving(), 0)