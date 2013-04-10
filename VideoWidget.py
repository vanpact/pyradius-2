#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    .. module:: VideoWidget
        :platform: Unix, Windows
        :synopsis: module which manage the video. It also contains a widget to visualize it.
    .. moduleauthor:: Yves-Rémi Van Eycke <yveycke [at] ulb.ac.be>
"""

"""
 /****************************************************************************
 **
 ** Copyright (C) 2011 Nokia Corporation and/or its subsidiary(-ies).
 ** All rights reserved.
 ** Contact: Nokia Corporation (qt-info@nokia.com)
 **
 ** This file has been created from an example of the Qt Toolkit in C++.
 **
 ** $QT_BEGIN_LICENSE:BSD$
 ** You may use this file under the terms of the BSD license as follows:
 **
 ** "Redistribution and use in source and binary forms, with or without
 ** modification, are permitted provided that the following conditions are
 ** met:
 **   * Redistributions of source code must retain the above copyright
 **     notice, this list of conditions and the following disclaimer.
 **   * Redistributions in binary form must reproduce the above copyright
 **     notice, this list of conditions and the following disclaimer in
 **     the documentation and/or other materials provided with the
 **     distribution.
 **   * Neither the name of Nokia Corporation and its Subsidiary(-ies) nor
 **     the names of its contributors may be used to endorse or promote
 **     products derived from this software without specific prior written
 **     permission.
 **
 ** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 ** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 ** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 ** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 ** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 ** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 ** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 ** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 ** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 ** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 ** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
 ** $QT_END_LICENSE$
 **
 ****************************************************************************/
 """
 
from PyQt4 import QtMultimedia, QtGui, QtCore
import cv2, numpy, ffvideo
from ffvideo import VideoStream
import gc, sys, debugsp


class VideoWidgetSurface(QtMultimedia.QAbstractVideoSurface):
    """ VideoWidgetSurface is a class used for the video presentation in a QWidget."""

    
    def __init__(self, parent=None):
        """Constructs the video surface. 
        
        Args:
            parent: A QWidget object representing the parent.
        """
        self.widget = None
        self.imageFormat = None
        self.targetRect = None
        self.imageSize = None
        self.sourceRect = None
        self.currentFrame = None
        if parent is None:
            super(VideoWidgetSurface, self).__init__()
        else: 
            super(VideoWidgetSurface, self).__init__(parent)
            self.widget = parent
        self.imageFormat = QtGui.QImage.Format_Invalid

        
    def supportedPixelFormats(self, handleType):
        """ Returns the supported pixel format.
            
        Args:
            handleType: A QtMultimedia.QAbstractVideoBuffer.HandleType object representing to identify the video buffers handle.
        
        Returns: 
            A list of QtMultimedia.QVideoFrame.PixelFormat containing the supported pixels formats. If No pixel formats are supported, the list will be empty.
        """
        if(handleType == QtMultimedia.QAbstractVideoBuffer.NoHandle ):
            return [QtMultimedia.QVideoFrame.Format_RGB32, QtMultimedia.QVideoFrame.Format_ARGB32, QtMultimedia.QVideoFrame.Format_RGB32_Premultiplied, QtMultimedia.QVideoFrame.Format_RGB565, QtMultimedia.QVideoFrame.Format_RGB555]
        else:
            return []
        
    def  isFormatSupported(self, form):
        """ Check if the video surface format is supported. The function will check if the image format is valid, has a size and if the HandleType is NoHandle.
        
        Args:
            form: A QtMultimedia.QVideoSurfaceFormat object to be checked.
        
        Returns: 
            True if the format is supported. False otherwise.
        """
        imageFormat=QtMultimedia.QVideoFrame.imageFormatFromPixelFormat(form.pixelFormat())
        size = form.frameSize()
        
        return (imageFormat != QtGui.QImage.Format_Invalid  and not size.isEmpty() and format.handleType() == QtMultimedia.QAbstractVideoBuffer.NoHandle)
        
    def present(self, frame):
        """ Force an immediate repaint of the video surface when a new frame has been received.
        Args:
            frame: A QtMultimedia.QVideoFrame object to be painted
            
        Returns:
            True if there was no error. False otherwise. 
        """
        if (self.surfaceFormat().pixelFormat() != frame.pixelFormat() or self.surfaceFormat().frameSize() != frame.size()):
            raise Exception
            self.stop()
            return False
        else: 
            del(self.currentFrame)
            self.currentFrame = frame
            del(frame)
            self.widget.repaint(self.targetRect)
            return True
        
    def start(self, form):
        """Start the video surface.
        
        Args: 
            form: The QtMultimedia.QVideoSurfaceFormat used by the surface. 
            
        Returns:
            True if there was no error. False otherwise. 
        """
        imageFormat = QtMultimedia.QVideoFrame.imageFormatFromPixelFormat(form.pixelFormat())
        size = form.frameSize()
        if (imageFormat != QtGui.QImage.Format_Invalid and not size.isEmpty()):
            self.imageFormat = imageFormat
            self.imageSize = size
            self.sourceRect = form.viewport()
            super(VideoWidgetSurface, self).start(form)
            self.widget.updateGeometry()
            self.updateVideoRect()
            return True
        else:
            return False
        
    def stop(self):
        """stop the video surface."""
        self.currentFrame = QtMultimedia.QVideoFrame();
        self.targetRect = QtCore.QRect();
        super(VideoWidgetSurface, self).stop();
        self.widget.update();
    
    def videoRect(self):
        """Returns a rectangle of the video.
        
        Returns: 
            A QtCore.QRect representing the painted zone.
        """
        return self.targetRect
        
    def updateVideoRect(self):
        """Rescale the rectangle to paint."""
        size = self.surfaceFormat().sizeHint()
        size.scale(self.widget.size().boundedTo(size), QtCore.Qt.KeepAspectRatio)
        self.targetRect = QtCore.QRect(QtCore.QPoint(0, 0), size)
        self.targetRect.moveCenter(self.widget.rect().center())
        
    def paint(self, painter):
        """Paints the current video frame on the widget surface.
        
        Args: 
            painter: A QtGui.QPainter used to draw the image on the surface.
        """
        if (self.currentFrame.map(QtMultimedia.QAbstractVideoBuffer.ReadOnly)):
            oldTransform = painter.transform()
             
        if (self.surfaceFormat().scanLineDirection() == QtMultimedia.QVideoSurfaceFormat.BottomToTop):
            painter.scale(1, -1)
            painter.translate(0, -self.widget.height())
        image= QtGui.QImage(self.currentFrame.bits(), self.currentFrame.width(), self.currentFrame.height(), self.currentFrame.bytesPerLine(), self.imageFormat)
        painter.drawImage(self.targetRect, image, self.sourceRect)
        painter.setTransform(oldTransform)
        self.currentFrame.unmap()
        del(oldTransform)
        del(image)
        del(painter)
            
class VideoWidget(QtGui.QWidget):
    """The VideoWidget class implement the actual videoWidget embedding the VideoWidgetSurface."""
    clicked = QtCore.pyqtSignal()
    clicking = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        """Constructs the video Widget. 
        
        Args:
            parent: A QWidget object representing the parent.
        """
        if parent is None:
            super(VideoWidget, self).__init__()
        else: 
            super(VideoWidget, self).__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_PaintOnScreen, True)
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Background, QtCore.Qt.black);
        self.setPalette(palette);
        self.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.surface = VideoWidgetSurface(self)
        self.LinesToDraw = True
        self.Lines = []
        self.PointsToDraw = True
        self.Points = []
        self.lastPoint = None
        self.stringsToDraw = False
        self.stringDrawn = []
        self.stringStartPosition = QtCore.QPoint(0, 0)
        self.mouseClick = False
          
    def videoSurface(self):
        """Return the VideoWidgetSurface.
        
        Returns:
            The VideoWidgetSurface embedded in the widget.
        """
        return self.surface
        
    def sizeHint(self): 
        """Return the recommend size for the widget.
        
        Returns:
            The recommend size for the widget.
        """
        return self.surface.surfaceFormat().sizeHint()
    
    def mouseReleaseEvent(self, e):
        if ((self.mouseClick) and (e.pos() == self.lastPoint)):
            self.clicked.emit()
    def mousePressEvent (self, e):
        self.lastPoint = e.pos()
        self.mouseClick = True
        self.clicking.emit()
    
    def appendLine(self, begin, end):
        if(isinstance(begin, QtCore.QPoint) and isinstance(end, QtCore.QPoint)):
            self.Lines.append((begin, end))
            self.repaint()
     
    def appendPoint(self, point):
        if(isinstance(point, QtCore.QPoint)):
            self.Points.append(point)
            self.repaint()
            
    def getLineNumbers(self):
        return len(self.Lines)
    
    def getPointNumbers(self):
        return len(self.Points)
                   
    def getLines(self):
        return self.Lines
    
    def getPoints(self):
        return self.Points
    
    def getString(self):
        return str(self.stringDrawn)
    
    def setString(self, string):
        self.stringsToDraw = True;
        self.stringDrawn = QtCore.QString(string)
        
    def getStringStartPosition(self):
        return (self.stringStartPosition.x(), self.stringStartPosition.y())
    
    def setStringStartPosition(self, position):
        self.stringStartPosition = QtCore.QPoint(position[0], position[1])
    
    def removeLastLine(self):
        self.Lines.pop()  
        self.repaint() 
        
    def popLastLine(self):
        l = self.Lines.pop()  
        self.repaint() 
        return l
    
    def stopDrawingString(self):
        self.stringDrawn = False
 
    def startDrawingString(self):
        self.stringDrawn = True
               
    def removeLastPoint(self):
        self.Points.pop()  
        self.repaint() 
          
    def popLastPoint(self):
        p = self.Points.pop()  
        self.repaint() 
        return p
    
    def resetShapes(self):
        self.Points = []
        self.Lines = []
    def paintEvent(self, event): 
        """Slot called when the widget receives a paint event.
        
        Args: 
            event: The received event.
        """
        painter= QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        if(self.surface.isActive()):
            videoRect = self.surface.videoRect()
            if (not (videoRect.contains(event.rect()))):
                region = event.region()
                region.subtract(QtGui.QRegion(videoRect))
                brush = self.palette().background()
                for rect in region.rects():
                    painter.fillRect(rect, brush)
                    del(rect)
                del(region)
            self.surface.paint(painter)
            del(videoRect)
        else: 
            painter.fillRect(event.rect(), self.palette().background())
        brush = QtGui.QBrush()
        brush.setColor(QtGui.QColor(255, 0, 0, 200))
        pen = QtGui.QPen(QtGui.QColor.green)
        pen.setWidth(3)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setColor(QtGui.QColor(255, 0, 0, 200))
        painter.setBrush(brush)
        painter.setPen(pen)
        if(self.PointsToDraw):
            for p in self.Points:
                painter.drawPoint(p)
        if(self.LinesToDraw):
            for l in self.Lines:
                painter.drawLine(l[0], l[1])
        font = QtGui.QFont()
        font.setPointSize(20)
        painter.setFont(font)
        if(self.stringsToDraw):
            painter.drawText(self.stringStartPosition, self.stringDrawn)
        del(painter)
        del(event)

    def resizeEvent(self, event):
        """Slot called when the widget is resized.
        
        Args: 
            event: The received event.
        """
        super(VideoWidget, self).resizeEvent(event)
        self.surface.updateVideoRect()

class Movie(QtCore.QObject):
    """This class represent the video data stream"""

    frameChanged = QtCore.pyqtSignal()
    endOfVideo = QtCore.pyqtSignal()
    
    def __init__(self, fileName=None):
        """Initialize the video stream and open the video file if a fileName is provided.
        
        Args: 
            filename: a string containing the name of the file to be opened.
        """
        self.rawBuffer=None
        self.source=None
        super(Movie, self).__init__()
        if(fileName is not None):
            self.setMovie(fileName)
        self.timer = None
        self.frame = None
        #    source = None
        #    imageBuffer = None
        #    rawBuffer = None
        self.isPlaying = False
        self.frameRate = 0
        self.frameNumber = 0
    
    def reset(self):
        self.rawBuffer=None
        self.source=None
        self.timer = None
        self.frame = None
        self.isPlaying = False
        self.frameRate = 0
        self.frameNumber = 0
    def setMovie(self, fileName):
        """Open a video file.
        
        Args:
            filename: a string containing the name of the file to be opened.
            
        raise:
            TypeError: The fileName is not a string.
        """
        if(isinstance(fileName, basestring)):
            self.source = VideoStream(str(fileName))
        else: 
            raise TypeError('fileName must be a string')
        
        self.frameRate = self.source.framerate
        self.frameNumber = self.source.duration*1000/self.source.framerate
#        if(self.frameRate==0):
#            self.source.set(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO, 1)
#            timeMax = float(self.source.get(cv2.cv.CV_CAP_PROP_POS_MSEC))/1000.0
#            if(self.frameNumber >= 1 and timeMax >= 1):
#                self.frameRate = float(self.frameNumber)/float(timeMax)
#            self.source.set(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO, 0)
#        if(self.frameRate==0):
#            self.frameRate = 25#Take a probable value
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000.0/self.frameRate)
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.frameMustChange)
        
    def play(self): 
        """Start to read the video stream."""
        self.isPlaying = True
        self.timer.start()
    
    def pause(self):
        """Pause the reading of the video stream."""
        self.isPlaying = False
        self.timer.stop()
        
    def frameMustChange(self):
        """Slot called when it is time to load the next frame.
        Raise: 
            Exception: The file cannot be read because the codec is not supported or the video is compressed.
        """
        self.readNextFrame()
        
#    def currentImage(self):
#        """Returns a QImage representing the image in the current frame.
#        
#        Returns: 
#            A QtGui.QImage representing the current frame.
#        """
#        return self.imageBuffer
    
#    def currentFrame(self):
#        """Returns the current frame.
#        
#        Returns: 
#            A QtMultimedia.QVideoFrame representing the current frame."""
#        return self.frame
    
#    def currentNdarrayFrame(self):
#        """Returns a NdArray representing the image in the current frame.
#        
#        Returns: 
#            A QtGui.QImage representing the current frame.
#        """
#        return self.rawBuffer
    
    def toggleState(self):
        """Toggle between playing and pausing the video."""
        if(self.isPlaying==True):
            self.pause()
        else:
            self.play()
            
    def jumpToFrame(self, position):
        """Modify the position in the video.
        
        Args: 
            position: a value between 0 and 1 corresponding to the position in the video. 0 is the beginning and 1 is the end.
        """
        if(position>1.0):
            position = 1.0
        elif(position<0.001):
            position = 0.001
        frame = self.source.get_frame_at_sec(position*self.source.duration).ndarray
        return frame
    
    def readNextFrame(self):
        """Load the next frame.
        Raise: 
            Exception: The file cannot be read because the codec is not supported or the video is compressed.
        """
        try:
            self.rawBuffer = self.source.next().ndarray()
        except ffvideo.NoMoreData:
            self.isPlaying = False
            self.pause()
            self.rawBuffer = None
            self.endOfVideo.emit()
#        if(error==0):
#            raise ValueError('Cannot read the file. Be sure the video is not compressed.')
#        imSize = image.shape
#        self.rawBuffer=numpy.copy(image)
#        image = numpy.ravel(image)
#        image.tostring()
#        image = QtGui.QImage(image, imSize[1], imSize[0], imSize[1]*3, QtGui.QImage.Format_RGB888)
#        self.frame = QtMultimedia.QVideoFrame(image)
#        del(imSize)
#        del(image)
        self.frameChanged.emit()
        
    def readNCFrame(self, number):
        """Load the next frame.
        Raise: 
            Exception: The file cannot be read because the codec is not supported or the video is compressed.
        """
        position = self.source.current().frameno
        try:
                self.rawBuffer = self.source.get_frame_no(position+number).ndarray()
        except ffvideo.NoMoreData:
            self.isPlaying = False
            self.pause()
            self.endOfVideo.emit()
        if(self.source.current().frameno>=self.source.duration*self.source.framerate):
            self.isPlaying = False
            self.pause()
            self.endOfVideo.emit()
#        error, image = self.source.read()
#        if(error==0):
#            raise ValueError('Cannot read the file. Be sure the video is not compressed.')
#        image = numpy.ravel(image)
#        image.tostring()
#        self.rawBuffer=image
#        image = QtGui.QImage(image, image.shape[1], image.shape[0], image.shape[1]*3, QtGui.QImage.Format_RGB888)
#        self.frame = QtMultimedia.QVideoFrame(image)
#        del(image)
        self.frameChanged.emit()

    def currentPositionRatio(self):
        """Returns the position in the video.
        
        Returns:
            a value between 0 and 1 representing the position in the video. 0 is the beginning and 1 is the end.
        """
        if(self.source is not None and self.source.current() is not None):
            return self.source.current().timestamp/self.source.duration
        else:
            return 1.0
    
    def getFrameNumber(self):
        """Returns the number of frame in the video.
        
        Returns:
            an integer representing the number of frame in the video.
        """
        return int(self.source.duration*self.source.framerate)
        
    