'''
Created on Oct 20, 2012

@author: yvesremi
'''
import cv2, numpy


class AbstractTreatment(object):
    def __init__(self):
        '''
        Constructor
        '''
        if(type(self) is AbstractTreatment):
            raise NotImplementedError('This class is abstract and cannot be instantiated') 
        super(AbstractTreatment, self).__init__()
        
    def compute(self, img):
            raise NotImplementedError( "The method need to be implemented" )
        
class blobDetectionTreatment(AbstractTreatment):
    def __init__(self, offset = [0, 0], mode = cv2.cv.CV_RETR_LIST, method=cv2.cv.CV_CHAIN_APPROX_NONE):
        self.mode = mode
        self.method = method
        self.xoffset = offset[0]
        self.yoffset = offset[1]

    def compute2(self, imgPre, img):
        contours, hierarchy = cv2.findContours(imgPre.copy(), mode=self.mode, method=self.method)
        datay = numpy.zeros((len(contours), 1), dtype=numpy.float32)
        datax = numpy.zeros((len(contours), 1), dtype=numpy.float32)
        dataa = numpy.zeros((len(contours), 1), dtype=numpy.float32)
        datavx = numpy.zeros((len(contours), 1), dtype=numpy.float32)
        datavy = numpy.zeros((len(contours), 1), dtype=numpy.float32)
        datan = numpy.zeros((len(contours), 1), dtype=numpy.float32)
        i=0
        for c in contours:
            xy = c[:,0,:]
            norm = numpy.square(numpy.max(xy, 0)-numpy.min(xy, 0))
            datan[i]=numpy.sqrt(norm[0]+norm[1])
            (vx, vy, x0, y0) = cv2.fitLine(points=xy, distType=cv2.cv.CV_DIST_L2, param=0, reps=10, aeps=0.01)
            #cv2.line(img, (x0+self.xoffset, y0+self.yoffset), (x0+self.xoffset+vx*20., y0+self.yoffset+vy*20.), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
            datay[i] = y0
            dataa[i] = numpy.arctan2(vy, vx)
            datax[i] = x0
            datavx[i] = vx
            datavy[i] = vy
            i = i + 1
        meanNorm = numpy.mean(datan)
#        meanvx = numpy.mean(datavx)
#        meanvy = numpy.mean(datavy)
        toKeep = numpy.all([datan>meanNorm*1.1], 0)
        datay = datay[toKeep]
        datax = datax[toKeep]
        dataa = dataa[toKeep]
        datavx = datavx[toKeep]
        datavy = datavy[toKeep]
        datan = datan[toKeep]
        retval, bestLabels, centers = cv2.kmeans(numpy.array((datay, dataa)).T, 3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10) , attempts=cv2.KMEANS_PP_CENTERS, flags=cv2.KMEANS_PP_CENTERS)
        mx0 = numpy.int(numpy.mean((numpy.atleast_2d(datax).T)[bestLabels==0]))
        my0 = numpy.int(numpy.mean((numpy.atleast_2d(datay).T)[bestLabels==0]))
        mvx0 = numpy.int(numpy.mean((numpy.atleast_2d(datavx).T)[bestLabels==0])*1000.0)
        mvy0 = numpy.int(numpy.mean((numpy.atleast_2d(datavy).T)[bestLabels==0])*1000.0)
        
        mx1 = numpy.int(numpy.mean((numpy.atleast_2d(datax).T)[bestLabels==1]))
        my1 = numpy.int(numpy.mean((numpy.atleast_2d(datay).T)[bestLabels==1]))
        mvx1 = numpy.int(numpy.mean((numpy.atleast_2d(datavx).T)[bestLabels==1])*1000.0)
        mvy1 = numpy.int(numpy.mean((numpy.atleast_2d(datavy).T)[bestLabels==1])*1000.0)
        
        mx2 = numpy.int(numpy.mean((numpy.atleast_2d(datax).T)[bestLabels==2]))
        my2 = numpy.int(numpy.mean((numpy.atleast_2d(datay).T)[bestLabels==2]))
        mvx2 = numpy.int(numpy.mean((numpy.atleast_2d(datavx).T)[bestLabels==2])*1000.0)
        mvy2 = numpy.int(numpy.mean((numpy.atleast_2d(datavy).T)[bestLabels==2])*1000.0)
#        for vx, vy, x0, y0 in zip((numpy.atleast_2d(datavx).T)[bestLabels==0], (numpy.atleast_2d(datavy).T)[bestLabels==0], (numpy.atleast_2d(datax).T)[bestLabels==0], (numpy.atleast_2d(datay).T)[bestLabels==0]):
#            cv2.line(img, (numpy.int(x0)+self.xoffset, numpy.int(y0)+self.yoffset), (numpy.int(x0)+self.xoffset+numpy.int(vx*20.), numpy.int(y0)+self.yoffset+numpy.int(vy*20.)), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)

#        for vx, vy, x0, y0 in zip((numpy.atleast_2d(datavx).T), (numpy.atleast_2d(datavy).T), (numpy.atleast_2d(datax).T), (numpy.atleast_2d(datay).T)):
#            cv2.line(img, (numpy.int(x0)+self.xoffset, numpy.int(y0)+self.yoffset), (numpy.int(x0)+self.xoffset+numpy.int(vx*20.), numpy.int(y0)+self.yoffset+numpy.int(vy*20.)), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)

        cv2.drawContours(img, contours, -1, (255, 255, 255))
        cv2.line(img, (mx0+self.xoffset, my0+self.yoffset), (mx0+self.xoffset+mvx0, my0+self.yoffset+mvy0), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
        cv2.line(img, (mx1+self.xoffset, my1+self.yoffset), (mx1+self.xoffset+mvx1, my1+self.yoffset+mvy1), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
        cv2.line(img, (mx2+self.xoffset, my2+self.yoffset), (mx2+self.xoffset+mvx2, my2+self.yoffset+mvy2), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
#        data = self.extractData(contours)
#        for c in contours
#        cv2.line(img, pt1, pt2, color)
        return img#self.normalizeData(data)
    
    def compute(self, imgPre, img):
        contours, hierarchy = cv2.findContours(imgPre.copy(), mode=self.mode, method=self.method)
        datac = numpy.zeros((len(contours), 2), dtype=numpy.float32)
        datas = numpy.ones((len(contours), 2), dtype=numpy.float32)
        dataa = numpy.zeros((len(contours), 1), dtype=numpy.float32)
        i=0
        for c in contours:
            if(len(c)>4):
                center, size, angle = cv2.fitEllipse(c)
                datac[i]=center
                datas[i]=size
                dataa[i]=angle
                #cv2.line(img, (x0+self.xoffset, y0+self.yoffset), (x0+self.xoffset+vx*20., y0+self.yoffset+vy*20.), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
            i = i + 1
#        meanvx = numpy.mean(datavx)
#        meanvy = numpy.mean(datavy)
        toKeep = [(datas[:, 1]/datas[:, 0])>2.0]
        datasm = numpy.mean(datas, 0)
        toKeep= [toKeep and datas[:, 1]*datas[:, 0]>(datasm[0]*datasm[1])*2] 
        dataa = numpy.radians(dataa[toKeep])
        datac=datac[toKeep]
        datas=datas[toKeep]
        retval, bestLabels, centers = cv2.kmeans(numpy.array((datac[:, 1], dataa)).T, 1, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10) , attempts=cv2.KMEANS_PP_CENTERS, flags=cv2.KMEANS_PP_CENTERS)
        mx0 = numpy.int(numpy.median((numpy.atleast_2d(datac[:, 0]).T)[bestLabels==0]))
        my0 = numpy.int(numpy.median((numpy.atleast_2d(datac[:, 1]).T)[bestLabels==0]))
        mvx0 = numpy.int(numpy.sin(numpy.median(dataa[bestLabels==0]))*1000.0)
        mvy0 = numpy.int(-numpy.cos(numpy.median(dataa[bestLabels==0]))*1000.0)
        
#        mx1 = numpy.int(numpy.mean((numpy.atleast_2d(datac[:, 0]).T)[bestLabels==1]))
#        my1 = numpy.int(numpy.mean((numpy.atleast_2d(datac[:, 1]).T)[bestLabels==1]))
#        mvx1 = numpy.int(numpy.sin(numpy.mean(dataa[bestLabels==1]))*1000.0)
#        mvy1 = numpy.int(-numpy.cos(numpy.mean(dataa[bestLabels==1]))*1000.0)
#        
#        mx2 = numpy.int(numpy.mean((numpy.atleast_2d(datac[:, 0]).T)[bestLabels==2]))
#        my2 = numpy.int(numpy.mean((numpy.atleast_2d(datac[:, 1]).T)[bestLabels==2]))
#        mvx2 = numpy.int(numpy.sin(numpy.mean(dataa[bestLabels==2]))*1000.0)
#        mvy2 = numpy.int(-numpy.cos(numpy.mean(dataa[bestLabels==2]))*1000.0)
#        for vx, vy, x0, y0 in zip((numpy.atleast_2d(datavx).T)[bestLabels==0], (numpy.atleast_2d(datavy).T)[bestLabels==0], (numpy.atleast_2d(datax).T)[bestLabels==0], (numpy.atleast_2d(datay).T)[bestLabels==0]):
#            cv2.line(img, (numpy.int(x0)+self.xoffset, numpy.int(y0)+self.yoffset), (numpy.int(x0)+self.xoffset+numpy.int(vx*20.), numpy.int(y0)+self.yoffset+numpy.int(vy*20.)), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)

#        for vx, vy, x0, y0 in zip(numpy.sin(dataa), -numpy.cos(dataa), (numpy.atleast_2d(datac[:, 0]).T), (numpy.atleast_2d(datac[:, 1]).T)):
#                        cv2.line(img, (numpy.int(x0)+self.xoffset, numpy.int(y0)+self.yoffset), (numpy.int(x0)+self.xoffset+numpy.int(vx*20.), numpy.int(y0)+self.yoffset+numpy.int(vy*20.)), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)

#        cv2.drawContours(img, contours, -1, (255, 255, 255))
        cv2.line(img, (mx0+self.xoffset-mvx0, my0+self.yoffset-mvy0), (mx0+self.xoffset+mvx0, my0+self.yoffset+mvy0), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
#        cv2.line(img, (mx1+self.xoffset, my1+self.yoffset), (mx1+self.xoffset+mvx1, my1+self.yoffset+mvy1), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
#        cv2.line(img, (mx2+self.xoffset, my2+self.yoffset), (mx2+self.xoffset+mvx2, my2+self.yoffset+mvy2), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
        return img#self.normalizeData(data)
    
class AponeurosisDetector(AbstractTreatment):
#   
    def __init__(self, offset = [0, 0], mode = cv2.cv.CV_RETR_LIST, method=cv2.cv.CV_CHAIN_APPROX_NONE):
        self.mode = mode
        self.method = method
        self.xoffset = offset[0]
        self.yoffset = offset[1]
    
    def compute(self, imgPre, img):
        contours, hierarchy = cv2.findContours(imgPre.copy(), mode=self.mode, method=self.method)
        datac = numpy.zeros((len(contours), 2), dtype=numpy.float32)
        datas = numpy.ones((len(contours), 2), dtype=numpy.float32)
        dataa = numpy.zeros((len(contours), 1), dtype=numpy.float32)
        i=0
        for c in contours:
            if(len(c)>4):
                center, size, angle = cv2.fitEllipse(c)
                datac[i]=center
                datas[i]=size
                dataa[i]=angle
                #cv2.line(img, (x0+self.xoffset, y0+self.yoffset), (x0+self.xoffset+vx*20., y0+self.yoffset+vy*20.), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
            i = i + 1
          
        goodLines = numpy.argsort((datas[:, 1]))#*datas[:, 1])/datas[:, 0])  
        for value in goodLines[::-1]:
            if(datac[value, 1]<=imgPre.shape[1]/4):
                largest = value
                break
#        largest=goodLines[len(goodLines)-1]
        dataa = numpy.radians(dataa)
        x0=datac[largest, 0]+self.xoffset
        y0=datac[largest, 1]+self.yoffset+datas[largest, 0]
        x1 = numpy.uint16(x0-numpy.sin(dataa[largest])*1000.0)
        y1 = numpy.uint16(y0+numpy.cos(dataa[largest])*1000.0)
        x2= numpy.uint16(x0+numpy.sin(dataa[largest])*1000.0)
        y2= numpy.uint16(y0-numpy.cos(dataa[largest])*1000.0)
#        cv2.drawContours(img, contours, -1, (255, 255, 255))
        cv2.line(img, (numpy.uint16(x1), numpy.uint16(y1)), (x2, y2), (255, 255, 255), 2)
#        cv2.ellipse(img, (datac[largest], datas[largest], dataao[largest]), (255, 255, 255), 2)

        for value in goodLines[::-1]:
            if(datac[value, 1]>imgPre.shape[1]/4):
                largest = value
                break
#        largest=goodLines[len(goodLines)-2]
        x0=datac[largest, 0]+self.xoffset
        y0=datac[largest, 1]+self.yoffset+datas[largest, 0]
        x1 = numpy.uint16(x0-numpy.sin(dataa[largest])*1000.0)
        y1 = numpy.uint16(y0+numpy.cos(dataa[largest])*1000.0)
        x2= numpy.uint16(x0+numpy.sin(dataa[largest])*1000.0)
        y2= numpy.uint16(y0-numpy.cos(dataa[largest])*1000.0)
#        cv2.drawContours(img, contours, -1, (255, 255, 255))
        cv2.line(img, (numpy.uint16(x1), numpy.uint16(y1)), (x2, y2), (255, 255, 255), 2)
        return img