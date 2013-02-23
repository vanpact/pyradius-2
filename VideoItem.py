'''
Created on Oct 4, 2012

@author: yvesremi
'''

from PyQt4 import QtMultimedia, QtGui, QtCore

class VideoItem(QtMultimedia.QAbstractVideoSurface, QtGui.QGraphicsItem):
    '''
    classdocs
    '''
    imageFormat = QtGui.QImage.Format_Invalid
    framePainted = False
    imageSize = None
    currentFrame = None
    

    def __init__(self, parent = None):
        '''
        Constructor
        '''
        if parent is None:
            super(VideoItem, self).__init__()
        else: 
            super(VideoItem, self).__init__(parent)
        
    def boundingRect(self):
        a=0
        a=a+1
    
    def paint(self, painter):
        if (self.currentFrame.map(QtMultimedia.QAbstractVideoBuffer.ReadOnly)):
            oldTransform = painter.transform()
             
        if (self.surfaceFormat().scanLineDirection() == QtMultimedia.QVideoSurfaceFormat.BottomToTop):
            painter.scale(1, -1)
            painter.translate(0, -self.widget.height())
        image= QtGui.QImage(self.currentFrame.bits(), self.currentFrame.width(), self.currentFrame.height(), self.currentFrame.bytesPerLine(), self.imageFormat)
        painter.drawImage(self.targetRect, image, self.sourceRect)
        painter.setTransform(oldTransform)
        self.framePainted = True
        self.currentFrame.unmap()

    def supportedPixelFormats(self, handleType):
        if(handleType == QtMultimedia.QAbstractVideoBuffer.NoHandle ):
            return [QtMultimedia.QVideoFrame.Format_RGB32, QtMultimedia.QVideoFrame.Format_ARGB32, QtMultimedia.QVideoFrame.Format_ARGB32_Premultiplied, QtMultimedia.QVideoFrame.Format_RGB24, QtMultimedia.QVideoFrame.Format_RGB565, QtMultimedia.QVideoFrame.Format_RGB555]
        else:
            return []
    
    def start(self, form):
        if (self.isFormatSupported(form)):
            self.imageFormat = QtMultimedia.QVideoFrame.imageFormatFromPixelFormat(form.pixelFormat())
            self.imageSize = form.frameSize()
            self.framePainted = True
            super(VideoItem, self).start(form)
            self.prepareGeometryChange()
            return True
        else:
            return False
        
    def stop(self):
        self.currentFrame = QtMultimedia.QVideoFrame();
        self.framePainted = False
        super(VideoItem, self).stop()

    def present(self, frame):
        if (not(self.framePainted)):
            if(not(self.isActive())):
                raise Exception('Video is not started!')
            return False
        else: 
            self.currentFrame = frame
            self.framePainted = False
            self.update()
            return True
        
    def sizeHint(self):
        return QtCore.QSize(800, 600)