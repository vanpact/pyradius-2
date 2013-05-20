#!/usr/bin/python
# -*- coding: utf-8 -*-

from PyQt4 import QtGui, QtCore
from VideoWidget import VideoWidget, Movie

class Tutorial(object):
    """Class for all tutorials"""
    def __init__(self, parent):
        """Constructor
        
        :param parent: The widget who asked to create this tutorial.
        :type parent: QWidget"""
        if(type(self) is Tutorial):
            raise NotImplementedError('This class is abstract and cannot be instantiated') 
        super(Tutorial, self).__init__()
        self.parent=parent
    
    def displayIntroduction(self):
        """Display to the user basic information of what he has to do"""
        if(type(self) is Tutorial):
            raise NotImplementedError('This method must be reimplemented') 
    
    def showConseils(self):
        """Display hints on how to place the extremities of the Aponeurosis"""
        box = QtGui.QMessageBox(QtGui.QMessageBox.Information, 'Conseils', 
'<p>Afin d\'obtenir de meilleurs résultats: \
<ul><li>Essayez de placer les droites correspondant aux aponévroses au centre de celles-ci. </li>\
<li>Les droites tracées pour les aponévroses ne doivent pas nécessairement aller d\'un bout à l\'autre de celles-ci. Si l\'aponévrose change beaucoup de courbure au cours du traitement, les résultats seront meilleurs avec de plus petites droites</li> \
<li>Afin d\'avoir de bons résultats, il est tout de même conseillé de ne pas tracer de droites de moins de 150 pixels</li></ul></p> \
', QtGui.QMessageBox.Ok, self.parent)
        box.show()
        
class JunctionTutorial(Tutorial):
    """Class for the tutorial specific to the treatment to extract the junction position."""
    def __init__(self, parent):
        """Constructor
        
        :param parent: The widget who asked to create this tutorial.
        :type parent: QWidget"""
        self.parent=parent
        super(LinesTutorial, self).__init__(self.parent)
        self.drawlist=['rectangle', 'point']
        
    def displayIntroduction(self):
        """Display to the user basic information of what he has to do"""
        tuto = QtGui.QMessageBox(QtGui.QMessageBox.Information, 'Information nécessaire', 
'<p>La méthode nécessite de connaitre la position de la région d\'intérêt et d\'avoir une première approximation pour la position de la jonction :\
<ul><li>Sur la vidéo, cliquez sur un point désignant un coin de la zone d\'intéret rectangulaire. Si rien n\'apparait, c\'est que vous avez mal cliqué.</li>\
<li>Si un point rouge est apparu à  l\'endroit où vous avez cliquez, cliquez sur le coin diagonalement opposé de la région d\'intéret. Un rectangle rouge devrait apparaitre à  l\'écran.</li>\
<li>Cliquez à  la position de la jonction.<\li></ul></p>\
<p><strong>N\'oubliez pas de modifier les options en bas de la fenêtre à  votre convenance avant de placer les aponévroses.</strong></p>\
', QtGui.QMessageBox.NoButton, self.parent)
        tutoLayout = tuto.layout()
        tutoWidget = tuto.layout().itemAtPosition(0, 1).widget()
        textMovieLayout = QtGui.QVBoxLayout()
        tutoLayout.addLayout(textMovieLayout, 0, 1)
        
        gifLabel = QtGui.QLabel()
        gif = QtGui.QMovie("Images/mouseSelectionJunction.gif", QtCore.QByteArray(), tuto)
        gif.setCacheMode(QtGui.QMovie.CacheAll)
        gif.setSpeed(100)
        gifLabel.setMovie(gif)
        
        textMovieLayout.addWidget(tutoWidget)
        textMovieLayout.addWidget(gifLabel)
        gifLabel.show()
        gif.start()
        tuto.addButton("Ok", QtGui.QMessageBox.AcceptRole)
        tuto.addButton("Conseils", QtGui.QMessageBox.HelpRole)
#         hintButton.clicked.connect(self.showConseils)
        ret = tuto.exec_()
        if(ret==1):
            self.showConseils()
        
class LinesTutorial(Tutorial):
    """Class for the tutorial specific to the treatment to extract the pennation angle using the ellipsoids detection or the Radon transform. """
    def __init__(self, parent):
        """Constructor
        
        :param parent: The widget who asked to create this tutorial.
        :type parent: QWidget"""
        self.parent=parent
        super(LinesTutorial, self).__init__(self.parent)
        self.drawlist=['line', 'line']
        
    def displayIntroduction(self):
        """Display to the user basic information of what he has to do"""
        tuto = QtGui.QMessageBox(QtGui.QMessageBox.Information, 'Information nécessaire', 
'<p>La méthode nécessite d\'une position initiale pour les deux aponévroses:\
<ul><li>Sur la vidéo, cliquez sur une extrémité au choix d\'une des deux aponévroses. Si rien n\'apparait, c\'est que vous avez mal cliqué.</li>\
<li>Si un point rouge est apparu à  l\'endroit où vous avez cliquez, cliquez sur l\'autre extrémité de la même aponévrose. Une ligne rouge devrait apparaitre à  l\'écran.</li> \
<li>Faites de même avec la deuxième aponévrose. </li></ul></p> \
<p><strong>N\'oubliez pas de modifier les options en bas de la fenêtre à  votre convenance avant de placer les aponévroses.</strong></p>\
', QtGui.QMessageBox.NoButton, self.parent)
        tutoLayout = tuto.layout()
        tutoWidget = tuto.layout().itemAtPosition(0, 1).widget()
        textMovieLayout = QtGui.QVBoxLayout()
        tutoLayout.addLayout(textMovieLayout, 0, 1)
        
        gifLabel = QtGui.QLabel()
        gif = QtGui.QMovie("Images/mouseSelection1.gif", QtCore.QByteArray(), tuto)
        gif.setCacheMode(QtGui.QMovie.CacheAll)
        gif.setSpeed(100)
        gifLabel.setMovie(gif)
        
        textMovieLayout.addWidget(tutoWidget)
        textMovieLayout.addWidget(gifLabel)
        gifLabel.show()
        gif.start()
        tuto.addButton("Ok", QtGui.QMessageBox.AcceptRole)
        tuto.addButton("Conseils", QtGui.QMessageBox.HelpRole)
#         hintButton.clicked.connect(self.showConseils)
        ret = tuto.exec_()
        if(ret==1):
            self.showConseils()
        
class MuscleTutorial(Tutorial):
    """Class for the tutorial specific to the treatment to extract the pennation angle using the Lucas-Kanade algorithm. """
    def __init__(self, parent):
        """Constructor
        
        :param parent: The widget who asked to create this tutorial.
        :type parent: QWidget"""
        self.parent=parent
        super(MuscleTutorial, self).__init__(self.parent)
        self.drawlist=['line', 'line', 'line']
        
    def displayIntroduction(self):
        """Display to the user basic information of what he has to do"""
        tuto = QtGui.QMessageBox(QtGui.QMessageBox.Information, 'Information nécessaire', 
'<p>La méthode nécessite une position initiale pour chacune des deux aponévroses et l\'orientation des fibres : \
<ul><li>Sur la vidéo, cliquez sur une extrémité au choix d\'une des deux aponévroses. Si rien n\'apparait, c\'est que vous avez mal cliqué.</li>\
<li>Si un point rouge est apparu à  l\'endroit où vous avez cliquez, cliquez sur l\'autre extrémité de la même aponévrose. Une ligne rouge devrait maintenant être à  l\'écran.</li> \
<li>Faites de même avec la deuxième aponévrose.</li> \
<li> Faites de même avec une des fibres. Il n\'est pas nécessaire d\'aller d\'une extremité à  l\'autre de la fibre puisque seule l\'orienation sera utilisée</li></ul></p> \
<p><strong>N\'oubliez pas de modifier les options en bas de la fenêtre à  votre convenance avant de placer les aponévroses.</strong></p>\
',QtGui.QMessageBox.NoButton, self.parent)
        tutoLayout = tuto.layout()
        tutoWidget = tuto.layout().itemAtPosition(0, 1).widget()
        textMovieLayout = QtGui.QVBoxLayout()
        tutoLayout.addLayout(textMovieLayout, 0, 1)
        
        gifLabel = QtGui.QLabel()
        gif = QtGui.QMovie("Images\mouseSelectionLK.gif", QtCore.QByteArray(), tuto)
        gif.setCacheMode(QtGui.QMovie.CacheAll)
        gif.setSpeed(100)
        gifLabel.setMovie(gif)
        
        textMovieLayout.addWidget(tutoWidget)
        textMovieLayout.addWidget(gifLabel)
        gifLabel.show()
        gif.start()
        tuto.addButton("Ok", QtGui.QMessageBox.AcceptRole)
        tuto.addButton("Conseils", QtGui.QMessageBox.HelpRole)
#         hintButton.clicked.connect(self.showConseils)
        ret = tuto.exec_()
        if(ret==1):
            self.showConseils()
        
        
