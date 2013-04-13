# -*- coding: utf-8 -*-

"""
    .. module:: View
        :platform: Unix, Windows
        :synopsis: module which manage the view of the application.
    .. moduleauthor:: Yves-Rémi Van Eycke <yveycke [at] ulb.ac.be>
"""

from PyQt4 import QtGui, QtCore, QtMultimedia
from VideoWidget import VideoWidget, Movie
import PreTreatments, Treatments, TotalTreatments #import Applier, CannyTreatment, GaborTreatment
from XlsWriter import XlsWriter
import gc, sys, debugsp


class Window(QtGui.QMainWindow):
    finishedWork = QtCore.pyqtSignal()
    """ This is the class which contains all the GUI."""
#    timeSlider = None
    
    def __init__(self):
        """ Create the view object."""
        super(Window, self).__init__()
        self.initUI()

    def initUI(self):
        """ Initialize the main windows of the application. """
        self.OutputFile = None
        self.resize(640, 480)
        self.centerWindow()
        self.setWindowTitle('Line')
        self.setWindowIcon(QtGui.QIcon('Images/Icon.png')) 
        QtCore.QTextCodec.setCodecForCStrings(QtCore.QTextCodec.codecForName('UTF-8'))
        
        self.firstTimePlayed = True
        self.createMenuBar()
        self.createLayout()
        self.source = None
        self.lines = []
        self.rect= []
#        self.source = movie()
#        QtCore.QObject.connect(self.source, QtCore.SIGNAL('frameChanged'), self.frameChanged)
#        self.source.frameChanged.connect(self.frameChanged)
#        self.filterApplied = PreTreatments.Applier()
#        self.filterApplied.frameComputed.connect(self.frameChanged)
        self.statusBar().showMessage('Ready')
        
    def centerWindow(self):
        """ Center the windows on the screen of the user. """
        windowsGeometry = self.frameGeometry()
        ScreenCenter = QtCore.QPoint(QtGui.QDesktopWidget().availableGeometry().center().x()/2, 0)
        windowsGeometry.moveCenter(ScreenCenter)
        self.move(windowsGeometry.center())       
     
    def createMenuBar(self):
        """ Create the Menu bar with its menu and add the actions to the menus. """
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&Fichiers')
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
        self.createInfoInterface()
        
        self.creatBasicControls()
        self.createOptionControls()
        
        mainLayout.addWidget(self.videoWidget)
        mainLayout.insertLayout(1, self.controlLayout)

        self.setCentralWidget(centralWidget)
        self.treatmentComboBox.currentIndexChanged.emit(0)
    
    def creatBasicControls(self):
        self.basicControlLayout = QtGui.QHBoxLayout()
        self.controlLayout.addLayout(self.basicControlLayout)
        
        self.playButton = QtGui.QPushButton('play')
        self.playButton.resize(50, 50)
        self.progressBar = QtGui.QProgressBar()
        self.progressBar.setOrientation(QtCore.Qt.Horizontal)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.playButton.setEnabled(False)        
        
        self.playButton.clicked.connect(self.toggleProcessVideo)
        
        self.basicControlLayout.addWidget(self.playButton)
        self.basicControlLayout.addWidget(self.progressBar)
        
    def createOptionControls(self):
        self.oLayout = QtGui.QHBoxLayout()
        self.controlLayout.addLayout(self.oLayout)
        
        self.OptionWidget = QtGui.QWidget(self)
        self.oLayout.addWidget(self.OptionWidget)
        
        
        self.OptionLayout = QtGui.QVBoxLayout()
        self.OptionWidget.setLayout(self.OptionLayout)
        self.OptionWidget.setContentsMargins(0, 0, 0, 0)
        self.OptionLayout.setContentsMargins(0, 0, 0, 0)
    
        self.basicOptionWidget = QtGui.QGroupBox("Options Générales", self.OptionWidget)
        self.OptionLayout.addWidget(self.basicOptionWidget)
        self.basicOptionLayout = QtGui.QHBoxLayout()
        self.basicOptionWidget.setLayout(self.basicOptionLayout)
        self.basicOptionWidget.setContentsMargins(0, 0, 0, 0)
#         self.basicOptionLayout.setContentsMargins(0, 0, 0, 0)

                
        self.treatmentComboBox = QtGui.QComboBox()
        self.treatmentComboBox.addItem('Lucas-Kanade')
        self.treatmentComboBox.addItem('Fit ellipsoïde')
        self.treatmentComboBox.addItem('Radon transform')
        self.treatmentComboBox.addItem('Seam Carving')
        self.outputLabel = QtGui.QLabel('Sauver les résultats :')
        self.outputCheckBox = QtGui.QCheckBox()
        
        self.SkipFrameLabel = QtGui.QLabel("Nombre d'images à passer:")
        self.SkipFrameLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.SkipFrameSpinBox = QtGui.QSpinBox()
        self.SkipFrameSpinBox.setMinimum(0)
        self.SkipFrameSpinBox.setMaximum(25)
        
        self.treatmentComboBox.currentIndexChanged.connect(self.loadCorrectInterface)
        self.outputCheckBox.stateChanged.connect(self.chooseOutputFile)
        
        self.basicOptionLayout.addStretch()
        self.basicOptionLayout.addWidget(self.treatmentComboBox)
        self.basicOptionLayout.addWidget(self.SkipFrameLabel)
        self.basicOptionLayout.addWidget(self.SkipFrameSpinBox)
        self.basicOptionLayout.addWidget(self.outputLabel)
        self.basicOptionLayout.addWidget(self.outputCheckBox)
        self.basicOptionLayout.addStretch()
        
        self.createEllipsoidMethodInterface()
        self.createRadonMethodInterface()
        self.createLKMethodInterface()
        self.createSCMethodInterface()
        
    def createInfoInterface(self):
        self.infoInterfaceLayout = QtGui.QHBoxLayout()
        self.controlLayout.addLayout(self.infoInterfaceLayout)
        self.infoLabelList = []
        self.infoValueList = []
        
    def updateInfoInterface(self, infoNumber):
        for value, label in zip(self.infoValueList, self.infoLabelList):
            del(value)
            del(label)
            
        for i in range(infoNumber):
            self.infoLabelList.append(QtGui.QLabel())
            self.infoValueList.append(QtGui.QLabel())
            self.infoInterfaceLayout.addWidget(self.infoLabelList[i])
            self.infoInterfaceLayout.addWidget(self.infoValueList[i])
            self.infoLabelList[i].show()
            self.infoValueList[i].show()
    
    def updateInfo(self, info):
        infoNumber = len(info)
        if(infoNumber!=len(self.infoValueList)):
            self.updateInfoInterface(infoNumber)
        for i, key in enumerate(info):
            self.infoLabelList[i].setText(key + " : ")
            self.infoValueList[i].setText(str(info[key]))
#         self.infoInterfaceWidget.update()

            
    def createEllipsoidMethodInterface(self):
        self.ellipsoidOptionWidget = QtGui.QGroupBox()
        self.ellipsoidOptionWidget.hide()
        self.OptionLayout.addWidget(self.ellipsoidOptionWidget)
        self.ellipsoidOptionLayout = QtGui.QHBoxLayout()
        self.ellipsoidOptionWidget.setLayout(self.ellipsoidOptionLayout)
        self.ellipsoidOptionLayout.addStretch()
        self.ellipsoidOptionWidget.setContentsMargins(0, 0, 0, 0)
#         self.ellipsoidOptionLayout.setContentsMargins(0, 0, 0, 0)

        
    def createRadonMethodInterface(self):
        self.radonOptionWidget = QtGui.QGroupBox()
        self.radonOptionWidget.hide()
        self.OptionLayout.addWidget(self.radonOptionWidget)
        self.radonOptionLayout = QtGui.QHBoxLayout()
        self.radonOptionWidget.setLayout(self.radonOptionLayout)
        self.multipleRadonLabel = QtGui.QLabel("Prendre plus d'échantillons:")
        self.multipleRadonLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.multipleRadonCheckBox = QtGui.QCheckBox()
        self.radonOptionLayout.addWidget(self.multipleRadonLabel)
        self.radonOptionLayout.addWidget(self.multipleRadonCheckBox)
        self.radonOptionLayout.addStretch()
        self.radonOptionWidget.setContentsMargins(0, 0, 0, 0)
#         self.radonOptionLayout.setContentsMargins(0, 0, 0, 0)
        
    def createLKMethodInterface(self):
        self.LKOptionWidget = QtGui.QGroupBox()
        self.LKOptionWidget.hide()
        self.OptionLayout.addWidget(self.LKOptionWidget)
        self.LKOptionLayout = QtGui.QHBoxLayout()
        self.LKOptionWidget.setLayout(self.LKOptionLayout)
        self.LKOptionLayout.addStretch()
        self.LKOptionWidget.setContentsMargins(0, 0, 0, 0)
#         self.LKOptionLayout.setContentsMargins(0, 0, 0, 0)
    
    def createSCMethodInterface(self):
        self.SCOptionWidget = QtGui.QGroupBox()
        self.SCOptionWidget.hide()
        self.OptionLayout.addWidget(self.SCOptionWidget)
        self.SCOptionLayout = QtGui.QHBoxLayout()
        self.SCOptionWidget.setLayout(self.SCOptionLayout)
        self.SCOptionWidget.setContentsMargins(0, 0, 0, 0)
#         self.SCOptionLayout.setContentsMargins(0, 0, 0, 0)
        self.SCThicknessLabel = QtGui.QLabel("épaisseur moyenne d'une aponévrose")
        self.SCThicknessSpinBox = QtGui.QSpinBox()
        self.SCThicknessSpinBox.setMinimum(0)
        self.SCThicknessSpinBox.setMaximum(200)
        self.SCThicknessSpinBox.setValue(45)
        self.SCOptionLayout.addWidget(self.SCThicknessLabel)
        self.SCOptionLayout.addWidget(self.SCThicknessSpinBox)
        self.SCOptionLayout.addStretch()
        
    def loadCorrectInterface(self):
        self.hideAllInterface()
        self.basicOptionWidget.show()
        if(self.treatmentComboBox.currentIndex()==0):
            self.loadLKTransformInterface()
        if(self.treatmentComboBox.currentIndex()==1):
            self.loadEllipsoidTransformInterface()
        if(self.treatmentComboBox.currentIndex()==2):
            self.loadRadonTransformInterface()
        if(self.treatmentComboBox.currentIndex()==3):
            self.loadSCTransformInterface()
       
    def hideAllInterface(self):
        for i in range(self.OptionLayout.count()):
            possibleWidget = self.OptionLayout.itemAt(i)
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
            fileName = QtCore.QString(QtGui.QFileDialog.getSaveFileName(self, "Ouvrir le fichier de sortie", QtCore.QDir.homePath(), 'Fichiers Excel (*.xls *.xlsx)'))
            if (not (fileName.isEmpty())):
                self.OutputFile = XlsWriter(unicode(fileName))
        else:
            self.openFile = None
                

             
    def openFile(self):
        """ Show a dialog where there user can pick a file. """
        fileName = QtCore.QString(QtGui.QFileDialog.getOpenFileName(self, "ouvrir la vidéo", QtCore.QDir.homePath(), 'Fichiers vidéo (*.avi *.mpeg *.mpg);;Autres (*.*)'))
        if (not (fileName.isEmpty())):
            self.surface.stop()
#            self.videoItem.stop()
            self.resetMovie()
            self.source.setMovie(str(fileName))
            self.filterApplied = PreTreatments.Applier()
            self.filterApplied.frameComputed.connect(self.frameChanged)
            self.filterApplied.setSource(self.source)
            self.filterApplied.endOfProcessing.connect(self.finishVideo)
            self.firstTimePlayed = True;
            self.playButton.setEnabled(False)
            self.toggleProcessVideo()
#            self.source.play()

    def resetMovie(self):
        if(self.source is not None):
            self.videoWidget.clicked.disconnect(self.getsurfacePosition)
            self.source.reset()
        else:
            self.source = Movie()
    def finishVideo(self):
        if(self.outputCheckBox.isChecked()):
            self.OutputFile.write(padNewColumn=True)
        self.treatmentComboBox.setEnabled(True)
        self.playButton.setEnabled(False)
        self.surface.stop()
        self.filterApplied.src=None
        self.source = Movie()
        self.firstTimePlayed = True;
        box = QtGui.QMessageBox(QtGui.QMessageBox.Information, 'Fin du Traitement', 
                          'Le traitement est terminé!', 
                          QtGui.QMessageBox.Ok, self)
        box.finished.connect(self.finishedWork.emit)
        box.show()
        
        
    def toggleProcessVideo(self):
        """Pause and play the video processing"""
        if(self.firstTimePlayed):
            self.firstTimePlayed = False
            self.filterApplied.applyOne()
            self.launchTutorial()
        else:
            self.playButton.setEnabled(True)
            self.filterApplied.toggle()
            if(self.filterApplied.wait):
                self.playButton.setText("Play")
            if(not(self.filterApplied.wait or self.filterApplied.isRunning())):
                    self.treatmentComboBox.setEnabled(False)
                    self.SkipFrameSpinBox.setEnabled(False)
                    self.outputCheckBox.setEnabled(False)
                    self.playButton.setText("Pause")
                    self.filterApplied.run()
#                     self.filterApplied.start(QtCore.QThread.HighestPriority)
            
        
    def frameChanged(self):
        """ Update the view when the frame has changed. """
#        frame = self.source.currentFrame()
        frame = self.filterApplied.getLastComputedFrame()
        info = self.filterApplied.getLastInformation()
        if(info is not None):
            self.updateInfo(info)
            if(self.outputCheckBox.isChecked()):
                self.OutputFile.addDatas(info)
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
        self.progressBar.setValue(int(self.source.currentPositionRatio()*self.progressBar.maximum()))
        del(frame)

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
#         p = self.videoWidget.mapToVideo(p)
        if(self.videoWidget.getPointNumbers() == 0):
            self.videoWidget.appendPoint(p)
        elif(not self.drawRect):
            p1 = self.videoWidget.popLastPoint()
            self.lines.append((p1, p))
            self.videoWidget.appendLine(p1, p)
        else:
            p1 = self.videoWidget.popLastPoint()
            self.rect.append((p1, p))
            self.videoWidget.appendRect(p1, p)     
                      
        if((self.videoWidget.getLineNumbers() >=self.lineNumber and not self.drawRect) or (self.drawRect and len(self.rect)>=1)):
            self.videoWidget.clicked.disconnect(self.getsurfacePosition)
            self.chooseTreatment(self.treatmentComboBox.currentIndex())
            self.videoWidget.resetShapes()
            self.toggleProcessVideo()
            
    def launchTutorial(self):
        self.treatmentComboBox.currentIndexChanged.connect(self.changeTutorial)
        self.drawRect = False
        self.lineNumber=0
        if(self.treatmentComboBox.currentIndex() == 0):
            self.tutorialmuscle()
        if(self.treatmentComboBox.currentIndex() == 1):
            self.tutorial1()
        if(self.treatmentComboBox.currentIndex() == 2):
            self.tutorial1()
        if(self.treatmentComboBox.currentIndex() == 3):
            self.tutorialjunction()
#             self.chooseTreatment(self.treatmentComboBox.currentIndex())
#             self.toggleProcessVideo()
    
    def changeTutorial(self):
        self.videoWidget.clicked.disconnect(self.getsurfacePosition)
        self.drawRect = False
        self.lineNumber=0
        self.videoWidget.resetShapes()
        self.lines = []
        self.rect = []
        if(self.treatmentComboBox.currentIndex() == 0):
            self.tutorialmuscle()
        if(self.treatmentComboBox.currentIndex() == 1):
            self.tutorial1()
        if(self.treatmentComboBox.currentIndex() == 2):
            self.tutorial1()
        if(self.treatmentComboBox.currentIndex() == 3):
            self.tutorialjunction()
#             self.chooseTreatment(self.treatmentComboBox.currentIndex())
#             self.toggleProcessVideo()
            
    def tutorialjunction(self):
        tuto = QtGui.QMessageBox(QtGui.QMessageBox.Information, 'Information nécessaire', 
'<p>La méthode nécessite de connaitre la position des bords de l’échographie :\
<ul><li>Sur la vidéo, cliquez sur un des coins de l’échographie au choix. Si rien n’apparait, c’est que vous avez mal cliqué.</li>\
<li>Si un point rouge est apparu à l’endroit où vous avez cliquez, cliquez sur le coin diagonalement opposé de l’échographie. Un rectangle rouge devrait apparaitre à l’écran.</li></ul></p>\
<p><strong>N’oubliez pas de modifier les options en bas de la fenêtre à votre convenance avant de placer les aponévroses.</strong></p>\
', QtGui.QMessageBox.Ok, self)
        tutoLayout = tuto.layout()
        tutoWidget = tuto.layout().itemAtPosition(0, 1).widget()
        textMovieLayout = QtGui.QVBoxLayout()
        tutoLayout.addLayout(textMovieLayout, 0, 1)
        
        gifLabel = QtGui.QLabel()
        gif = QtGui.QMovie("mouseSelectionJunction.gif", QtCore.QByteArray(), tuto)
        gif.setCacheMode(QtGui.QMovie.CacheAll)
        gif.setSpeed(100)
        gifLabel.setMovie(gif)
        
        textMovieLayout.addWidget(tutoWidget)
        textMovieLayout.addWidget(gifLabel)
        gifLabel.show()
        gif.start()
        tuto.show()
        self.drawRect = True
        self.videoWidget.clicked.connect(self.getsurfacePosition)
        
    def tutorial1(self):
        tuto = QtGui.QMessageBox(QtGui.QMessageBox.Information, 'Information nécessaire', 
'<p>La méthode nécessite d\'une position initiale pour les deux aponévroses:\
<ul><li>Sur la vidéo, cliquez sur une extrémité au choix d’une des deux aponévroses. Si rien n’apparait, c’est que vous avez mal cliqué.</li>\
<li>Si un point rouge est apparu à l’endroit où vous avez cliquez, cliquez sur l’autre extrémité de la même aponévrose. Une ligne rouge devrait apparaitre à l’écran.</li> \
<li>Faites de même avec la deuxième aponévrose. </li></ul></p> \
<p><strong>N’oubliez pas de modifier les options en bas de la fenêtre à votre convenance avant de placer les aponévroses.</strong></p>\
', QtGui.QMessageBox.Ok, self)
        tutoLayout = tuto.layout()
        tutoWidget = tuto.layout().itemAtPosition(0, 1).widget()
        textMovieLayout = QtGui.QVBoxLayout()
        tutoLayout.addLayout(textMovieLayout, 0, 1)
        
        gifLabel = QtGui.QLabel()
        gif = QtGui.QMovie("mouseSelection1.gif", QtCore.QByteArray(), tuto)
        gif.setCacheMode(QtGui.QMovie.CacheAll)
        gif.setSpeed(100)
        gifLabel.setMovie(gif)
        
        textMovieLayout.addWidget(tutoWidget)
        textMovieLayout.addWidget(gifLabel)
        gifLabel.show()
        gif.start()
        tuto.show()
        self.lineNumber = 2
        self.videoWidget.clicked.connect(self.getsurfacePosition)
        
    def tutorialmuscle(self):
        tuto = QtGui.QMessageBox(QtGui.QMessageBox.Information, 'Information nécessaire', 
'<p>La méthode nécessite une position initiale pour chacune des deux aponévroses et l\'orientation des fibres : \
<ul><li>Sur la vidéo, cliquez sur une extrémité au choix d’une des deux aponévroses. Si rien n’apparait, c’est que vous avez mal cliqué.</li>\
<li>Si un point rouge est apparu à l’endroit où vous avez cliquez, cliquez sur l’autre extrémité de la même aponévrose. Une ligne rouge devrait maintenant être à l’écran.</li> \
<li>Faites de même avec la deuxième aponévrose.</li> \
<li> Faites de même avec une des fibres. Il n\'est pas nécessaire d\'aller d\'une extremité à l\'autre de la fibre puisque seule l\'orienation sera utilisée</li></ul></p> \
<p><strong>N’oubliez pas de modifier les options en bas de la fenêtre à votre convenance avant de placer les aponévroses.</strong></p>\
', QtGui.QMessageBox.Ok, self)
        tutoLayout = tuto.layout()
        tutoWidget = tuto.layout().itemAtPosition(0, 1).widget()
        textMovieLayout = QtGui.QVBoxLayout()
        tutoLayout.addLayout(textMovieLayout, 0, 1)
        
        gifLabel = QtGui.QLabel()
        gif = QtGui.QMovie("mouseSelectionLK.gif", QtCore.QByteArray(), tuto)
        gif.setCacheMode(QtGui.QMovie.CacheAll)
        gif.setSpeed(100)
        gifLabel.setMovie(gif)
        
        textMovieLayout.addWidget(tutoWidget)
        textMovieLayout.addWidget(gifLabel)
        gifLabel.show()
        gif.start()
        tuto.show()
        self.lineNumber = 3
        self.videoWidget.clicked.connect(self.getsurfacePosition)
                
    def chooseTreatment(self, index):
        """Add the filters to be applied."""
        if(index == 0):
            self.filterApplied.setParameters(self.source, nrSkipFrame = self.SkipFrameSpinBox.value())
            l = self.videoWidget.mapToVideo(self.videoWidget.getLines())
            self.filterApplied.setMethod(TotalTreatments.LKMethod(Aponeurosises=l[0:2], fiber=l[2]))
        if(index == 1):
            self.filterApplied.setParameters(self.source, nrSkipFrame = self.SkipFrameSpinBox.value())
            l = self.videoWidget.mapToVideo(self.videoWidget.getLines())
            self.filterApplied.setMethod(TotalTreatments.EllipseMethod(Aponeurosises = l))
        if(index == 2):
            self.filterApplied.setParameters(self.source, nrSkipFrame = self.SkipFrameSpinBox.value())
            l = self.videoWidget.mapToVideo(self.videoWidget.getLines())
            self.filterApplied.setMethod(TotalTreatments.RadonMethod(Aponeurosises = l, manySamples=self.multipleRadonCheckBox))
        if(index == 3):
            self.filterApplied.setParameters(self.source,  nrSkipFrame = self.SkipFrameSpinBox.value())
            r = self.videoWidget.mapToVideo(self.videoWidget.getRect())[0]
            self.filterApplied.setMethod(TotalTreatments.junctionComputation(limits=r, AponeurosisThickness=self.SCThicknessSpinBox.value()))