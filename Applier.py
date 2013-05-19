
import gc
from PyQt4 import QtCore, QtMultimedia
from VideoWidget import Movie
from imageConverter import ImageConverter
from ResultsWriter import ResultsWriter

class Applier(QtCore.QThread):
    """This is the class which apply all the filters. For the preTreatment and the treatments."""
    frameComputed = QtCore.pyqtSignal()
    """Qt signal indicacting a frame has been computer"""
    endOfProcessing = QtCore.pyqtSignal()
    """Qt signal indicacting the end of the processing"""
    frameToSend = None
    """The computed frame. workaround for a problem with memory allocation."""
    
    def __init__(self, src=None, nrSkipFrame=0):
        """    
        :param src: The video to process
        :type src: Movie
        :param nrSkipFrame: The number of frame to skip
        :type nrSkipFrame: Int
        
        Constructor. 
        """
#            Applier.__single = Applier.__single
        super(Applier, self).__init__()
#        Applier.__single.filtersToPreApply = []
#        Applier.__single.filtersToPreApply = [ [] for i in range(nrChannel)]
        self.backToFirst=False
        self.methodToApply = None
        self.processedVideo = []
        self.lastComputedFrame = None
        self.wait=True
        self.angleFinal=0
        self.nrSkipFrame=nrSkipFrame
        self.infoGathered = []
        self.writeFile='output.txt'
        self.src = None
        self.setSource(src)
    
    def setParameters(self, src=None, nrSkipFrame=0):
        """    
        :param src: The video to process
        :type src: Movie
        :param nrSkipFrame: The number of frame to skip
        :type nrSkipFrame: Int
        
        Set the parameters for the applier.
        """
        del(self.processedVideo)
        self.processedVideo = []
        self.infoGathered = []
        self.setSource(src)
        self.nrSkipFrame=nrSkipFrame
        
    def __del__(self):
        """    
        Destructor    
        """
        del(self.methodToApply)
        del(self.processedVideo)
        if(self.src is not None):
            del(self.src)
        del(self.writeFile)
        
    def toggle(self):
        """Toggle the applier state. if wait is true, the applier will pause the processing of the video."""
        self.wait = not self.wait
        
          
    def setSource(self, src=file):
        """    
        :param src: The video to process
        :type src: Movie
        
        Set the source of the video.
        """
        if(isinstance(src, Movie)):
            self.wait=True
            self.processedVideo = []
            if(self.src is not None):
                self.src.endOfVideo.disconnect(self.finishedVideo)
                del(self.src)
            self.src = src
            self.src.endOfVideo.connect(self.finishedVideo)
        elif src is not None:
            raise TypeError("Src must be none or a Movie!")
        
    def setMethod(self, method):
        """    
        :param method: The treatment to apply
        :type method: Treatment
        
        Set the treatment to apply to the video.  
        """
        self.methodToApply=method
        
    def apply(self, img):
        """    
        :param img: The img to process
        :type img: Numpy array 
        :return: the processed image
        :rtype: Numpy array
        
        Process the image and collect the results for further use.
        """
        img, info = self.methodToApply.compute(img)
        info[str('Time')] = self.src.getEllapsedTime()
        self.infoGathered.append(info)
        gc.collect()
        return img

            
    def applyNext(self):
        """    
        Select the next frame on which the method has to be applied.
        """
        if(self.backToFirst==True):
            self.backToFirst=False
            self.src.readNCFrame(0)
        else:
            self.src.readNCFrame(self.nrSkipFrame+1)
        if(self.src is not None and self.src.rawBuffer is not None):
            ndimg = self.apply(self.src.rawBuffer)
            del(self.lastComputedFrame)
            self.lastComputedFrame=ndimg
            del(ndimg)
#        self.processedVideo.append(ndimg)
            self.frameComputed.emit()
#        return QtMultimedia.QVideoFrame(ImageConverter.ndarrayToQimage(ndimg))
        
    def applyOne(self):
        """    
        Apply the method on the first frame of the video. 
        """
        self.src.readNCFrame(0)
        self.backToFirst=True
        ndimg = self.src.rawBuffer
        del(self.lastComputedFrame)
        self.lastComputedFrame=ndimg
        del(ndimg)
#        self.processedVideo.append(ndimg)
        self.frameComputed.emit()
#        return QtMultimedia.QVideoFrame(ImageConverter.ndarrayToQimage(ndimg))
        
    def applyAll(self):
        """    
        Apply the method on the whole video. 
        """
        frameNb = self.src.getFrameNumber()
        self.finished=False
        while(not self.finished):
            self.applyNext()
            while(self.wait):
                self.msleep(50)
    
    def run(self):
        """    
        Method necessary for the multi-threading with Qt.
        """
        self.applyAll()
#         self.finished=True
#         self.endOfProcessing.emit()
    
    def finishedVideo(self):
        """    
        Method called when the wall video has been processed. Emit a Qt signal to inform the video has been processed.
        """
        self.finished=True
        self.endOfProcessing.emit()
        
    def getLastComputedFrame(self):
        """    
        :return: the frame
        :rtype: QVideoFrame
        
        Get the last processed frame.
        """
        self.frameToSend = ImageConverter.ndarrayToQimage(self.lastComputedFrame)#self.processedVideo[-1]).copy()
        return QtMultimedia.QVideoFrame(self.frameToSend)

    def getLastComputedImage(self):
        """    
        :return: the image
        :rtype: QImage
        
        Get the last processed image.
        """
#         self.frameToSend = ImageConverter.ndarrayToQimage(self.lastComputedFrame)#self.processedVideo[-1]).copy()
        return self.frameToSend
    
    def getLastInformation(self):
        """    
        :return: the information
        :rtype: Dictionnary
        
        Get the information extracted from the last processed frame.
        """
        if(len(self.infoGathered)>0):
            return self.infoGathered[-1]
        else:
            return None
    
    def saveResult(self, inFileName=None, outFileName=None):
        """    
        Dump all the information extracted in a file.
        """
        writer = ResultsWriter(inFileName)
        writer.addDatas(self.infoGathered)
        writer.write(outFileName=outFileName)