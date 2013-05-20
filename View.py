#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    .. module:: View
        :platform: Unix, Windows
        :synopsis: module which manage the view of the application.
    .. moduleauthor:: Yves-Rémi Van Eycke <yveycke [at] ulb.ac.be>
"""

from PyQt4 import QtGui, QtCore, QtMultimedia
from VideoWidget import VideoWidget, Movie
import Tutorial
from Applier import Applier
import TotalTreatments #import Applier, CannyTreatment, GaborTreatment


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
        self.point = []
        self.OutputFile=None
        self.InputFile=None
#        self.source = movie()
#        QtCore.QObject.connect(self.source, QtCore.SIGNAL('frameChanged'), self.frameChanged)
#        self.source.frameChanged.connect(self.frameChanged)
#        self.filterApplied = PreTreatments.Applier()
#        self.filterApplied.frameComputed.connect(self.frameChanged)
        self.statusBar().showMessage('Prêt')
        
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
        fileMenu.addAction(self.createAction('&Fermer', QtGui.qApp.quit, 'Ctrl+Q', 'Quitte application'))
        fileMenu.addAction(self.createAction('&Ouvrir', self.openFile, 'Ctrl+O', 'Ouvre un film'))
        self.printImageAction=self.createAction('&Sauver image', self.captureImage, 'Ctrl+S', 'Sauvegarde l\'image présente à l\'écran')
        fileMenu.addAction(self.printImageAction)
        self.printImageAction.setDisabled(True)
        helpMenu = menubar.addMenu('Aide')
        helpMenu.addAction(self.createAction('A propos', self.printAbout, 'Ctrl+?', 'A propos'))
        helpMenu.addAction(self.createAction('A propos de Qt', self.printAboutQt, 'A propos de qt'))
        
    def printAbout(self):
        """Display the information about the application"""
        QtGui.QMessageBox.about(self, 'A propos de Pyradius', '<p><strong>Pyradius V1.0</strong></p>\
        <p>Pyradius est un logiciel de traitement d\'images permettant l\'extraction de l\'angle de pennation et de la position de la jonction des aponévroses.</p>\
        <p>Copyright&copy; 2013 Yves-Rémi Van Eycke.</p>\
        <p>Ce logiciel est sous licence <a href=https://www.gnu.org/licenses/quick-guide-gplv3.fr.html>GNU GPLv3</a>.\
        Il a été rendu possible grâce à: \
        <a href=\'http://www.python.org/\'>Python</a>, <a href=\'http://www.cython.org/\'>Cython</a>, \
        <a href=\'http://www.riverbankcomputing.co.uk/software/pyqt\'>PyQt</a>, <a href=\'http://www.numpy.org/\'>Numpy</a>, \
        <a href=\'http://opencv.org/\'>OpenCV</a>, <a href=\'http://scikit-image.org/\'>scikit-image</a>, \
        <a href=\'https://bitbucket.org/zakhar/ffvideo/wiki/Home\'>FFVideo</a>, <a href=\'http://www.ffmpeg.org/\'>FFMpeg</a>, \
        <a href=\'http://pandas.pydata.org/\'>Pandas</a>.</p>\
        Le code source est disponible <a href=\'https://code.google.com/p/pyradius/\'>ici</a>.<p></p>')
        
    def printAboutQt(self):
        """Display the information about Qt"""
        QtGui.QMessageBox.aboutQt(self, 'A propos de Qt')
        
    def createAction(self, name, funct, shortcut='', statusTip=''):
        """Create the actions used, for example, in the menu bar and return it.
        
        :param name: The name of the action.
        :param funct: The function to be triggered by the action.
        :param shortcut: The shortcut for the action.
        :param statusTip: A tip which will appear when hovering the action.
        :return: The QAction object
        :raise: AttributeError: if there is the name or the function send is none.
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
        self.mainLayout = QtGui.QVBoxLayout(centralWidget)
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
        
        self.mainLayout.addWidget(self.videoWidget)
        self.mainLayout.insertLayout(1, self.controlLayout)

        self.setCentralWidget(centralWidget)
        self.treatmentComboBox.currentIndexChanged.emit(0)
    
    def creatBasicControls(self):
        """ Create the layout for the widgets controlling the video. """
        self.basicControlLayout = QtGui.QHBoxLayout()
        self.controlLayout.addLayout(self.basicControlLayout)
        
        self.playButton = QtGui.QPushButton(QtGui.QIcon("Images/play.png"), 'Commencer')
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
        """ Create the layout for the widgets common to all the treatments. """
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

        self.resetTreatmentButton = QtGui.QPushButton("Réinitialiser")
        self.resetTreatmentButton.setEnabled(False)
        
        self.treatmentComboBox = QtGui.QComboBox()
        self.treatmentComboBox.addItem('Lucas-Kanade')
        self.treatmentComboBox.addItem('Fit ellipsoïde')
        self.treatmentComboBox.addItem('Radon transform')
        self.treatmentComboBox.addItem('Seam Carving')
#         self.outputLabel = QtGui.QLabel('Sauver les résultats :')
#         self.outputCheckBox = QtGui.QCheckBox()
        
        self.SkipFrameLabel = QtGui.QLabel("Nombre d'images à passer:")
        self.SkipFrameLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.SkipFrameSpinBox = QtGui.QSpinBox()
        self.SkipFrameSpinBox.setMinimum(0)
        self.SkipFrameSpinBox.setMaximum(25)
        
        self.resetTreatmentButton.clicked.connect(self.changeTutorial)
        self.treatmentComboBox.currentIndexChanged.connect(self.loadCorrectInterface)
#         self.outputCheckBox.stateChanged.connect(self.chooseOutputFile)
        
        self.basicOptionLayout.addStretch()
        self.basicOptionLayout.addWidget(self.resetTreatmentButton)
        self.basicOptionLayout.addWidget(self.treatmentComboBox)
        self.basicOptionLayout.addWidget(self.SkipFrameLabel)
        self.basicOptionLayout.addWidget(self.SkipFrameSpinBox)
#         self.basicOptionLayout.addWidget(self.outputLabel)
#         self.basicOptionLayout.addWidget(self.outputCheckBox)
        self.basicOptionLayout.addStretch()
        
        self.createEllipsoidMethodInterface()
        self.createRadonMethodInterface()
        self.createLKMethodInterface()
        self.createSCMethodInterface()
        
    def createInfoInterface(self):
        """ Create the layout for the labels displaying the extracted information. """
        self.infoInterfaceLayout = QtGui.QHBoxLayout()
        self.controlLayout.addLayout(self.infoInterfaceLayout)
        self.infoLabelList = []
        self.infoValueList = []
        
    def updateInfoInterface(self, infoNumber):
        """ Update the layout containing the labels displaying the extracted information. 
        
        :param infoNumber: The number of info to display (=the number of labels needed).
        :type infoNumber: int
        """
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
        """ Update the labels containing the extracted information. 
        
        :param info: The new extracted information.
        :type info: dictionary
        """
        infoNumber = len(info)
        if(infoNumber!=len(self.infoValueList)):
            self.updateInfoInterface(infoNumber)
        for i, key in enumerate(info):
            self.infoLabelList[i].setText(key + " : ")
            self.infoValueList[i].setText(str(info[key]))
#         self.infoInterfaceWidget.update()

            
    def createEllipsoidMethodInterface(self):
        """ Create the layout for the widgets specific to the treatment which use ellipsoids. """
        self.ellipsoidOptionWidget = QtGui.QGroupBox()
        self.ellipsoidOptionWidget.hide()
        self.OptionLayout.addWidget(self.ellipsoidOptionWidget)
        self.ellipsoidOptionLayout = QtGui.QHBoxLayout()
        self.ellipsoidOptionWidget.setLayout(self.ellipsoidOptionLayout)
        self.ellipsoidOptionLayout.addStretch()
        self.ellipsoidOptionWidget.setContentsMargins(0, 0, 0, 0)
#         self.ellipsoidOptionLayout.setContentsMargins(0, 0, 0, 0)

        
    def createRadonMethodInterface(self):
        """ Create the layout for the widgets specific to the treatment which use the Radon transform. """
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
        """ Create the layout for the widgets specific to the treatment which use Lucas-Kanade algorithm. """
        self.LKOptionWidget = QtGui.QGroupBox()
        self.LKOptionWidget.hide()
        self.OptionLayout.addWidget(self.LKOptionWidget)
        self.LKOptionLayout = QtGui.QHBoxLayout()
        self.LKOptionWidget.setLayout(self.LKOptionLayout)
        self.LKOptionLayout.addStretch()
        self.LKOptionWidget.setContentsMargins(0, 0, 0, 0)
#         self.LKOptionLayout.setContentsMargins(0, 0, 0, 0)
    
    def createSCMethodInterface(self):
        """ Create the layout for the widgets specific to the treatment extraction the junction position. """
        self.SCOptionWidget = QtGui.QGroupBox()
        self.SCOptionWidget.hide()
        self.OptionLayout.addWidget(self.SCOptionWidget)
        self.SCOptionLayout = QtGui.QHBoxLayout()
        self.SCOptionWidget.setLayout(self.SCOptionLayout)
        self.SCOptionWidget.setContentsMargins(0, 0, 0, 0)
#         self.SCOptionLayout.setContentsMargins(0, 0, 0, 0)
        
    def loadCorrectInterface(self):
        """ Select the correct layout depending on the method chosen by the user. """
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
        """ Hide all the layouts containing the treatment specific widgets. """
        for i in range(self.OptionLayout.count()):
            possibleWidget = self.OptionLayout.itemAt(i)
            if(possibleWidget.widget()):
                possibleWidget.widget().hide()
    
    def loadEllipsoidTransformInterface(self):
        """ Load the layout for the widgets specific to the treatment which use ellipsoids. """
        self.ellipsoidOptionWidget.show()
        
    def loadRadonTransformInterface(self):
        """ Create the layout for the widgets specific to the treatment which use the Radon transform. """
        self.radonOptionWidget.show()
    
    def loadLKTransformInterface(self):
        """ Create the layout for the widgets specific to the treatment which use Lucas-Kanade algorithm. """
        self.LKOptionWidget.show()
    
    def loadSCTransformInterface(self):
        """ Create the layout for the widgets specific to the treatment extraction the junction position. """
        self.SCOptionWidget.show()
        
    def chooseOutputFile(self):  
        """ Show the dialog to let the user select the output file. """
        
        fileName = QtCore.QString(QtGui.QFileDialog.getSaveFileName(self, "Ouvrir le fichier de sortie", QtCore.QDir.homePath(), 'Fichiers Excel (*.xlsx *.xls);;Fichiers Texte (*.txt);;Fichiers CSV (*.csv)'))
        if (not (fileName.isEmpty())):
            self.OutputFile = (unicode(fileName))

    def chooseInputFile(self):  
        """ Show the dialog to let the user select the output file. """
        fileName = QtCore.QString(QtGui.QFileDialog.getOpenFileName(self, "Ouvrir le fichier d'entrée", QtCore.QDir.homePath(), 'Fichiers Excel (*.xlsx *.xls);;Fichiers Texte (*.txt);;Fichiers CSV (*.csv)'))
        if (not(fileName.isEmpty())):
            self.InputFile = (unicode(fileName))           

             
    def openFile(self):
        """ Show a dialog where there user can pick a file. """
        fileName = QtCore.QString(QtGui.QFileDialog.getOpenFileName(self, "ouvrir la vidéo", QtCore.QDir.homePath(), 'Fichiers vidéo (*.avi *.mpeg *.mpg);;Autres (*.*)'))
        if (not (fileName.isEmpty())):
            self.surface.stop()
#            self.videoItem.stop()
            self.resetMovie()
            self.source.setMovie(unicode(fileName))
            self.filterApplied = Applier()
            self.filterApplied.frameComputed.connect(self.frameChanged)
            self.filterApplied.setSource(self.source)
            self.filterApplied.endOfProcessing.connect(self.finishVideo)
            self.firstTimePlayed = True;
            self.playButton.setEnabled(False)
            self.toggleProcessVideo()
            self.printImageAction.setEnabled(True)
            self.resetTreatmentButton.setEnabled(True)
#            self.source.play()

    def resetMovie(self):
        """ Reset the movie. """
        if(self.source is not None):
            self.videoWidget.clicked.disconnect(self.getsurfacePosition)
            self.source.reset()
        else:
            self.source = Movie()
    def finishVideo(self):
        """ Close the video, ask the applier to write the output file if necessary and quit. """
        self.statusBar().showMessage('Traitement terminé')
        box = QtGui.QMessageBox(QtGui.QMessageBox.Information, 'Sauvegarde', 
                  '<p><strong>Le traitement est terminé! Voulez vous sauver les résultats?</strong></p> \
                  <p>Si vous sélectionnez oui, deux fenêtres de dialogue vont s\'ouvrir. \
                  <ul><li>La première va permettre de choisir un fichier d\'entrée si vous voulez fusionner les résultats avec d\'autres résultats que vous possèdez déjà. Dans le cas où vous ne désirez pas fusionner de résultats, fermez cette fenêtre sans choisir de fichier.</li>\
                  <li>La deuxième va permettre de choisir où les résultats seront sauvegardés.</li></ul></p>', 
                  QtGui.QMessageBox.Yes|QtGui.QMessageBox.No, self)
        self.treatmentComboBox.setEnabled(True)
        self.resetTreatmentButton.setEnabled(True)
        self.playButton.setEnabled(False)
        self.surface.stop()
        ret = box.exec_()
        if(ret==QtGui.QMessageBox.Yes):
            self.savingProcedure()
        box = QtGui.QMessageBox(QtGui.QMessageBox.Information, 'Fin du traitement', 
            'Travail terminé! Le programme va maintenant se fermer', 
            QtGui.QMessageBox.Ok, self)
        box.finished.connect(self.finishedWork.emit)
        box.buttonClicked.connect(self.finishedWork.emit)
        self.statusBar().showMessage('Fermeture du programme')
        box.show()
        
    def savingProcedure(self):
        """Save the results in a file"""
        cont=True
        while(cont):
            if(self.InputFile is None or not self.InputFile):
                self.chooseInputFile()
            if(self.OutputFile is None or not self.OutputFile):
                self.chooseOutputFile()
            if(self.OutputFile is None or not self.OutputFile):
                box = QtGui.QMessageBox(QtGui.QMessageBox.Information, 'Pas de fichier de sortie', 
                  'Vous n\'avez pas sélectionné de fichier de sortie. Les resultats ne seront donc pas sauvegardés. Êtes vous sûr de votre choix?', 
                  QtGui.QMessageBox.Yes|QtGui.QMessageBox.No, self)
                ret = box.exec_()
                if(ret==QtGui.QMessageBox.Yes):
                    cont=False
                else:
                    continue
            if(cont):
                self.statusBar().showMessage('Sauvegarde en cours')
                self.filterApplied.saveResult(self.InputFile, self.OutputFile)
                cont=False
                self.statusBar().showMessage('Sauvegarde terminée')
        
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
                self.playButton.setIcon(QtGui.QIcon("Images/play.png"))
                self.playButton.setText("Continuer")
            if(not(self.filterApplied.wait) or not(self.filterApplied.isRunning())):
                    self.treatmentComboBox.setEnabled(False)
                    self.resetTreatmentButton.setEnabled(False)
                    self.SkipFrameSpinBox.setEnabled(False)
                    self.playButton.setIcon(QtGui.QIcon("Images/pause.png"))
                    self.playButton.setText("Pause")
#                    self.filterApplied.run()
                    self.filterApplied.start(QtCore.QThread.HighestPriority)
            
        
    def frameChanged(self):
        """ Update the view when the frame has changed. """
        frame = self.filterApplied.getLastComputedFrame()
        info = self.filterApplied.getLastInformation()
        if(info is not None):
            self.updateInfo(info)

        if (not(frame.isValid())):
            QtGui.QMessageBox(QtGui.QMessageBox.Critical, 'Error', 'Frame not valid!', QtGui.QMessageBox.Ok, self).show()
            return False;
        currentFormat = self.surface.surfaceFormat()
        if (frame.pixelFormat() != currentFormat.pixelFormat() or frame.size() != currentFormat.frameSize()):
            fmt = QtMultimedia.QVideoSurfaceFormat(frame.size(), frame.pixelFormat())
            if (not(self.surface.start(fmt))):
                QtGui.QMessageBox(QtGui.QMessageBox.Critical, 'Error', 'Surface could not start!', QtGui.QMessageBox.Ok, self).show()
                return False
        if (not(self.surface.present(frame))):
            self.surface.stop()
        self.progressBar.setValue((self.source.currentPositionRatio()*self.progressBar.maximum()))
        del(frame)

    def jumpToFrame(self, value):
        """ Jump to a position in the movie. 
        
        :param value: A value in the range of the slider.
        :raise: AttributeError: if value is to high or negative.
        """
        ratio = float(value)/float(self.progressBar.maximum())
        if(ratio<=1.0 and ratio>=0.0):
            self.source.jumpToFrame(ratio)
        else:
            raise ValueError("Trying to jump at a position which does not exists")
         
    def captureImage(self):
        """ Take a snapshot of the video and save it. """
#        snapshot = self.source.currentImage()
        snapshot = self.filterApplied.getLastComputedImage()
        fileName = QtGui.QFileDialog.getSaveFileName(self, "save an image", QtCore.QDir.homePath(), 'Image (*.png *.PNG)')
        snapshot.save(fileName, "png")
#         self.source.play()
        
    def getsurfacePosition(self):
        """Get the position of the cursor in the video and ask the surface to draw a shape if necessary."""
        p=QtGui.QCursor.pos()
        p=self.videoWidget.mapFromGlobal(p)
        shape=self.drawlist[0]
        if(shape=='point' or self.videoWidget.getPointNumbers() == 0):
            self.videoWidget.appendPoint(p) 
            if(shape=='point'):
                self.drawlist.pop(0)
        elif(shape=='rectangle' and self.videoWidget.getPointNumbers() == 1):
            p1 = self.videoWidget.popLastPoint()
            self.videoWidget.appendRect(p1, p)  
            self.drawlist.pop(0)
        elif(shape=='line' and self.videoWidget.getPointNumbers() == 1):
            p1 = self.videoWidget.popLastPoint()
            self.videoWidget.appendLine(p1, p)
            self.drawlist.pop(0)                  
        if(len(self.drawlist)==0):
            self.videoWidget.clicked.disconnect(self.getsurfacePosition)
            self.chooseTreatment(self.treatmentComboBox.currentIndex())
            self.videoWidget.resetShapes()
            self.statusBar().showMessage('Traitement en cours')
            self.toggleProcessVideo()
            
    def launchTutorial(self):
        """Launch a tutorial specific to the selected treatment."""
        self.statusBar().showMessage('En attente d\'informations de la part de l\'utilisateur')
        self.treatmentComboBox.currentIndexChanged.connect(self.changeTutorial)
        self.drawlist=[]
        self.tutorial=None
        if(self.treatmentComboBox.currentIndex() == 0):
            tutorial = Tutorial.MuscleTutorial(self)
        if(self.treatmentComboBox.currentIndex() == 1):
            tutorial = Tutorial.LinesTutorial(self)
        if(self.treatmentComboBox.currentIndex() == 2):
            tutorial = Tutorial.LinesTutorial(self)
        if(self.treatmentComboBox.currentIndex() == 3):
            tutorial = Tutorial.JunctionTutorial(self)
        self.drawlist=tutorial.drawlist
        tutorial.displayIntroduction()
        self.videoWidget.clicked.connect(self.getsurfacePosition)
#             self.chooseTreatment(self.treatmentComboBox.currentIndex())
#             self.toggleProcessVideo()
    
    def changeTutorial(self):
        """Reset the shapes drawn on the surface and reload a new tutorial specific to the selected treatment."""
        self.statusBar().showMessage('En attente d\'informations de la part de l\'utilisateur')
        self.videoWidget.clicked.disconnect(self.getsurfacePosition)
        self.drawlist=[]
        self.videoWidget.resetShapes()
        self.tutorial=None
        if(self.treatmentComboBox.currentIndex() == 0):
            tutorial = Tutorial.MuscleTutorial(self)
        if(self.treatmentComboBox.currentIndex() == 1):
            tutorial = Tutorial.LinesTutorial(self)
        if(self.treatmentComboBox.currentIndex() == 2):
            tutorial = Tutorial.LinesTutorial(self)
        if(self.treatmentComboBox.currentIndex() == 3):
            tutorial = Tutorial.JunctionTutorial(self)
        self.drawlist=tutorial.drawlist
        tutorial.displayIntroduction()
        self.videoWidget.clicked.connect(self.getsurfacePosition)
#             self.chooseTreatment(self.treatmentComboBox.currentIndex())
#             self.toggleProcessVideo()
            
                
    def chooseTreatment(self, index):
        """Add the chosen treatment to the Applier.
        
        :param index: The index inf the combobox used to select the treatment.
        :type index: int
        """
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
            self.filterApplied.setMethod(TotalTreatments.RadonMethod(Aponeurosises = l, manySamples=self.multipleRadonCheckBox.isChecked()))
        if(index == 3):
            self.filterApplied.setParameters(self.source,  nrSkipFrame = self.SkipFrameSpinBox.value())
            r = self.videoWidget.mapToVideo(self.videoWidget.getRect())[0]
            p = self.videoWidget.mapToVideo(self.videoWidget.getPoints())[0]
            self.filterApplied.setMethod(TotalTreatments.junctionComputation(limits=r, firstApproximation=p))