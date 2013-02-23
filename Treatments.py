'''
Created on Oct 20, 2012

@author: yvesremi
'''
import cv2, numpy
import skimage.transform

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
 
#@jit       
class blobDetectionTreatment(AbstractTreatment):
    
#    @void(uint16[:], uint8, uint8)
    def __init__(self, offset = [0, 0], mode = cv2.cv.CV_RETR_LIST, method=cv2.cv.CV_CHAIN_APPROX_NONE):
        self.mode = mode
        self.method = method
        self.xoffset = offset[0]
        self.yoffset = offset[1]
        self.previousAngle = 2.0
        
#    @autojit   
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
#        toKeep = [(datas[:, 1]/datas[:, 0])>4.0]
        datasm = numpy.median(datas, 0)
#        toKeep= numpy.all([toKeep and (datas[:, 1]*datas[:, 0]>(datasm[0]*datasm[1])*10)], 0)
        toKeep = numpy.all([datas[:, 0]>0.0,(datas[:, 1]/datas[:, 0])>5.0, (datas[:, 1]*datas[:, 0]>(datasm[0]*datasm[1])*7.0)], 0)
        dataa = numpy.radians(dataa[toKeep])
        datac=datac[toKeep]
        datas=datas[toKeep]
#        retval, bestLabels, centers = cv2.kmeans(numpy.array((datac[:, 1], dataa)).T, 1, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10) , attempts=cv2.KMEANS_PP_CENTERS, flags=cv2.KMEANS_PP_CENTERS)
        mx0 = 0#numpy.int(numpy.median((numpy.atleast_2d(datac[:, 0]).T)))
        my0 = 0#numpy.int(numpy.median((numpy.atleast_2d(datac[:, 1]).T)))
        newAngle=numpy.float(numpy.median(dataa))
        if(not numpy.isnan(newAngle)):
            self.previousAngle = 2.0*self.previousAngle/5.0 + 3.0*newAngle/5.0
        mvx0 = numpy.int(numpy.sin(self.previousAngle)*5000.0)
        mvy0 = numpy.int(-numpy.cos(self.previousAngle)*5000.0)
        
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
        cv2.putText(img, "angle = " + numpy.str(numpy.degrees(self.previousAngle))[:6], org=(numpy.uint32(img.shape[0]*0.75), numpy.uint32(img.shape[1]*0.75)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2, bottomLeftOrigin=False)
        for value in range(toKeep.shape[0]):
            if(toKeep[value]):
                cv2.drawContours(img, contours[value], -1, (255, 255, 255), 2, offset=(self.xoffset, self.yoffset))
#            else:
#                cv2.drawContours(img, contours[value], -1, (128, 128, 128), 2, offset=(self.xoffset, self.yoffset))
        cv2.line(img, (mx0+self.xoffset-mvx0, my0+self.yoffset-mvy0), (mx0+self.xoffset+mvx0, my0+self.yoffset+mvy0), cv2.cv.Scalar(255, 0, 0), 2, cv2.CV_AA, 0)
#        cv2.line(img, (mx1+self.xoffset, my1+self.yoffset), (mx1+self.xoffset+mvx1, my1+self.yoffset+mvy1), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
#        cv2.line(img, (mx2+self.xoffset, my2+self.yoffset), (mx2+self.xoffset+mvx2, my2+self.yoffset+mvy2), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
        return img#self.normalizeData(data)
    
class AponeurosisDetector(AbstractTreatment):
#   
    def __init__(self, offset = [0, 0], mode = cv2.cv.CV_RETR_LIST, method=cv2.cv.CV_CHAIN_APPROX_NONE, angle=0):
        self.mode = mode
        self.method = method
        self.xoffset = offset[0]
        self.yoffset = offset[1]
        self.angle = angle

#    @jit(argtypes=[uint8[:,:], uint8[:,:]], restype=uint8[:, :])    
    def compute(self, imgPre, img):
        center = (imgPre.shape[0]/2.0, imgPre.shape[1]/2.0)
        M = cv2.getRotationMatrix2D(center, numpy.degrees(self.angle), 1.0)
        alpha = numpy.cos(-self.angle)
        beta = numpy.sin(-self.angle)
        sizemaxX = numpy.int(numpy.abs(alpha*imgPre.shape[1]))#-beta*img.shape[0])
        sizemaxY = numpy.int(numpy.abs(beta*imgPre.shape[1]+alpha*imgPre.shape[0]))
        imgPre = cv2.warpAffine(imgPre, M, (sizemaxX, sizemaxY))#[sizemaxY/2-30:sizemaxY/2+30, 0:sizemaxX]
        
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
#                cv2.drawContours(img, c, -1, (200, 200, 200), 2, offset=(self.xoffset, self.yoffset))
                cv2.ellipse(img, ((center[0]+self.xoffset, center[1]+self.yoffset), size, angle), (255, 255, 255), 1)
                #cv2.line(img, (x0+self.xoffset, y0+self.yoffset), (x0+self.xoffset+vx*20., y0+self.yoffset+vy*20.), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
            i = i + 1
          
        goodLines = numpy.argsort((datas[:, 1]))#*datas[:, 1])/datas[:, 0])  

        for value in goodLines[::-1]:
#            if(datac[value, 1]<=imgPre.shape[1]/4):
            if(True):
                largest = value
                cv2.drawContours(img, contours[value], -1, (200, 200, 200), 2, offset=(self.xoffset, self.yoffset))
#                cv2.drawContours(img, contours[value], -1, (255, 255, 255), 2, offset=(self.xoffset, self.yoffset))
                break
#        largest=goodLines[len(goodLines)-1]
        dataa = numpy.radians(dataa-90)
        dataax = numpy.cos(dataa[largest])+numpy.cos(self.angle)
        dataay = numpy.sin(dataa[largest])+numpy.sin(self.angle)
#        dataa[largest] = (dataa[largest]%numpy.pi+self.angle%numpy.pi)%numpy.pi
        dataatot = numpy.arctan(dataay/dataax)
        x0=datac[largest, 0]
        y0=datac[largest, 1]
        x1 = numpy.uint16(x0-dataax)*10
        y1 = numpy.uint16(y0-dataay)*10
        x2= numpy.uint16(x0+dataax)*10
        y2= numpy.uint16(y0+dataay)*10
#        cv2.drawContours(img, contours, -1, (255, 255, 255))
        cv2.line(img, (x1+self.xoffset, y1+self.yoffset), (x2+self.xoffset, y2+self.yoffset), (255, 255, 255), 2)
#        cv2.ellipse(img, ((datac[largest][0]+self.xoffset,datac[largest][1]+self.yoffset), datas[largest], -dataa[largest]+90), (255, 255, 255), 2)
        self.angle = dataatot
        for value in goodLines[::-1]:
            if(numpy.abs(numpy.int16(datac[value, 1])-numpy.int16(y0))>imgPre.shape[1]/8):
                largest = value
                cv2.drawContours(img, contours[value], -1, (200, 200, 200), 2, offset=(self.xoffset, self.yoffset))
                break
#        largest=goodLines[len(goodLines)-2]
        x0=datac[largest, 0]
        y0=datac[largest, 1]
        x1 = numpy.uint16(x0-numpy.sin(dataa[largest])*1000.0)
        y1 = numpy.uint16(y0+numpy.cos(dataa[largest])*1000.0)
        x2= numpy.uint16(x0+numpy.sin(dataa[largest])*1000.0)
        y2= numpy.uint16(y0-numpy.cos(dataa[largest])*1000.0)
#        cv2.drawContours(img, contours, -1, (255, 255, 255))
#        cv2.line(img, (x1+self.xoffset, y1+self.yoffset), (x2+self.xoffset, y2+self.yoffset), (255, 255, 255), 2)
#        cv2.ellipse(img, ((datac[largest][0]+self.xoffset,datac[largest][1]+self.yoffset), datas[largest], -dataa[largest]+90), (255, 255, 255), 2)
        return img
    
class LineFitTreatment(AbstractTreatment):
    def __init__(self, offset = [0, 0], mode = cv2.cv.CV_RETR_LIST, method=cv2.cv.CV_CHAIN_APPROX_NONE):
        self.mode = mode
        self.method = method
        self.xoffset = offset[0]
        self.yoffset = offset[1]
        self.previousAngle = 2.0

#    @jit(argtypes=[uint8[:,:], uint8[:,:]], restype=uint8[:, :])  
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
#                cv2.ellipse(img, ((center[0]+self.xoffset, center[1]+self.yoffset), size, angle), (255, 255, 255), 1)
                #cv2.line(img, (x0+self.xoffset, y0+self.yoffset), (x0+self.xoffset+vx*20., y0+self.yoffset+vy*20.), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
            i = i + 1
#        meanvx = numpy.mean(datavx)
#        meanvy = numpy.mean(datavy)
#        toKeep = [(datas[:, 1]/datas[:, 0])>4.0]
        datasm = numpy.median(datas, 0)
#        toKeep= numpy.all([toKeep and (datas[:, 1]*datas[:, 0]>(datasm[0]*datasm[1])*10)], 0)
        toKeep = numpy.all([datas[:,1]>datasm[1]*5.0, datas[:, 0]>0.0,(datas[:, 1]/datas[:, 0]>(datasm[1]/datasm[0])*7.0)], 0)
        dataa = numpy.radians(dataa[toKeep])
        datac=datac[toKeep]
        datas=datas[toKeep]
#        retval, bestLabels, centers = cv2.kmeans(numpy.array((datac[:, 1], dataa)).T, 1, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10) , attempts=cv2.KMEANS_PP_CENTERS, flags=cv2.KMEANS_PP_CENTERS)
        mx0 = 0#numpy.int(numpy.median((numpy.atleast_2d(datac[:, 0]).T)))
        my0 = 0#numpy.int(numpy.median((numpy.atleast_2d(datac[:, 1]).T)))
        newAngle=numpy.float(numpy.median(dataa))
        if(not numpy.isnan(newAngle)):
            self.previousAngle = 2.0*self.previousAngle/5.0 + 3.0*newAngle/5.0
        mvx0 = numpy.int(numpy.sin(self.previousAngle)*5000.0)
        mvy0 = numpy.int(-numpy.cos(self.previousAngle)*5000.0)
        
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
        cv2.putText(img, "angle = " + numpy.str(numpy.degrees(self.previousAngle))[:6], org=(numpy.uint32(img.shape[0]*0.75), numpy.uint32(img.shape[1]*0.75)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2, bottomLeftOrigin=False)
        for value in range(toKeep.shape[0]):
            if(toKeep[value]):
                cv2.drawContours(img, contours[value], -1, (255, 255, 255), 2, offset=(self.xoffset, self.yoffset))
#            else:
#                cv2.drawContours(img, contours[value], -1, (128, 128, 128), 2, offset=(self.xoffset, self.yoffset))
        cv2.line(img, (mx0+self.xoffset-mvx0, my0+self.yoffset-mvy0), (mx0+self.xoffset+mvx0, my0+self.yoffset+mvy0), cv2.cv.Scalar(255, 0, 0), 2, cv2.CV_AA, 0)
#        cv2.line(img, (mx1+self.xoffset, my1+self.yoffset), (mx1+self.xoffset+mvx1, my1+self.yoffset+mvy1), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
#        cv2.line(img, (mx2+self.xoffset, my2+self.yoffset), (mx2+self.xoffset+mvx2, my2+self.yoffset+mvy2), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
        return img#self.normalizeData(data)


class testRadon(AbstractTreatment):

    def __init__(self, offset = [0, 0]):
        self.xoffset = offset[0]
        self.yoffset = offset[1]

    def compute(self, imgPre, img):
        mask = numpy.zeros_like(imgPre)
        radius=min(numpy.int(imgPre.shape[0]/2), numpy.int(imgPre.shape[1]/2))
        centerx = numpy.int(imgPre.shape[1]/2)
        centery = numpy.int(imgPre.shape[0]/2)
        cv2.circle(img=mask, center=(centerx, centery), radius=radius, color=255, thickness=-1, lineType=8, shift=0)
#        imgPrefiltered = numpy.where(imgPre>=1, 1, 0)
        imgPrefiltered = numpy.where(mask>=1, imgPre, mask)
        imgRad = imgPrefiltered[centery-radius:centery+radius,centerx-radius:centerx+radius].copy()
        angleRange = range(0, 180, 5)
        angle = numpy.radians(angleRange[numpy.argmax(numpy.var(skimage.transform.radon(numpy.float64(imgPrefiltered), numpy.asarray(angleRange)), 0))])
        mvx0 = numpy.int(numpy.sin(angle)*5000.0)
        mvy0 = numpy.int(numpy.cos(angle)*5000.0)
#        cv2.line(img, (self.xoffset-mvx0, self.yoffset-mvy0), (self.xoffset+mvx0, self.yoffset+mvy0), cv2.cv.Scalar(255, 0, 0), 2, cv2.CV_AA, 0)
        cv2.line(img, (self.xoffset+imgPrefiltered.shape[1]/2-mvx0, self.yoffset+imgPrefiltered.shape[0]/2-mvy0), (self.xoffset+imgPrefiltered.shape[1]/2+mvx0, self.yoffset+imgPrefiltered.shape[0]/2+mvy0), cv2.cv.Scalar(255, 0, 0), 2, cv2.CV_AA, 0)
        cv2.circle(img=img, center=(numpy.int(self.xoffset+imgPrefiltered.shape[1]/2), numpy.int(self.yoffset+imgPrefiltered.shape[0]/2)), radius=min(numpy.int(imgPrefiltered.shape[0]/2), numpy.int(imgPrefiltered.shape[1]/2)), color=255, thickness=1, lineType=8, shift=0)
        imgPrefiltered = cv2.copyMakeBorder(imgPrefiltered, self.yoffset, img.shape[0]-(imgPrefiltered.shape[0]+self.yoffset), self.xoffset, img.shape[1]-(imgPrefiltered.shape[1]+self.xoffset), borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        imgPre = cv2.copyMakeBorder(imgPre, self.yoffset, img.shape[0]-(imgPre.shape[0]+self.yoffset), self.xoffset, img.shape[1]-(imgPre.shape[1]+self.xoffset), borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img=numpy.where(imgPrefiltered>0, imgPre, img)
        cv2.putText(img, "angle = " + numpy.str(numpy.degrees(angle))[:6], org=(numpy.uint32(img.shape[0]*0.75), numpy.uint32(img.shape[1]*0.75)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2, bottomLeftOrigin=False)
        return img
        
class aponeurosisHough(AbstractTreatment):
    
    def __init__(self, offset=[0, 0], angle=0):
        self.xoffset = offset[0]
        self.yoffset = offset[1]
        self.angle = angle
        
    def compute(self, imgPre, img):

        center = (imgPre.shape[0]/2.0, imgPre.shape[1]/2.0)
        M = cv2.getRotationMatrix2D(center, numpy.degrees(self.angle), 1.0)
        alpha = numpy.cos(-self.angle)
        beta = numpy.sin(-self.angle)
        sizemaxX = numpy.int(alpha*imgPre.shape[1])#-beta*img.shape[0])
        sizemaxY = numpy.int(beta*imgPre.shape[1]+alpha*imgPre.shape[0])
        imgPre = cv2.warpAffine(imgPre, M, (sizemaxX, sizemaxY))[sizemaxY/2-30:sizemaxY/2+30, 0:sizemaxX]
        
        lines = cv2.HoughLinesP(image = numpy.copy(numpy.uint8(imgPre)), rho=15, theta=numpy.float(cv2.cv.CV_PI)/180.0, threshold = 2, minLineLength = 100, maxLineGap = 7)
        lines = numpy.squeeze(lines)
        if(len(lines.shape)>1):
            angle = numpy.zeros(lines.shape[0], dtype=numpy.float32)
            leng = numpy.zeros(lines.shape[0], dtype=numpy.float32)
            i=0
            for line in lines:
                leng[i] = numpy.sqrt(numpy.power(line[3]-line[1], 2)+numpy.power(line[2]-line[0], 2))
                if(line[2]-line[0]!=0):
                    angle[i] = numpy.arctan(numpy.float(line[3]-line[1])/numpy.float(line[2]-line[0]))
                elif(line[3]-line[1]>0):
                    angle[i]=cv2.cv.CV_PI/2.0
                else:
                    angle[i]=-cv2.cv.CV_PI/2.0
                i=i+1
            sortedIndex = numpy.argsort(leng)
            n = numpy.ceil(sortedIndex.shape[0]/5.0)
            n = min(sortedIndex.shape[0], 3)
            valueNumber = sortedIndex.shape[0]-n
            accumx1 = 0
            accumy1 = 0
            accumx2 = 0
            accumy2 = 0
            accumAngleX = 0
            accumAngleY = 0
            for values in sortedIndex[valueNumber:]:
                accumAngleX = accumAngleX+numpy.cos(angle[values])
                accumAngleY = accumAngleY+numpy.sin(angle[values])
#                accumx1 = accumx1+lines[values, 0]
#                accumy1 = accumy1+lines[values, 1]
#                accumx2 = accumx2+lines[values, 2]
#                accumy2 = accumy2+lines[values, 3]
#                accumAngle = accumAngle+(angle[values]%cv2.cv.CV_PI)
#            accumx1 = accumx1/valueNumber
#            accumy1 = accumy1/valueNumber
#            accumx2 = accumx2/valueNumber*100
#            accumy2 = accumy2/valueNumber*100
            accumAngleX = (accumAngleX/n + numpy.cos(self.angle))*100
            accumAngleY =  (accumAngleY/n + numpy.sin(self.angle))*100
#            accumAngle = (self.angle%cv2.cv.CV_PI + accumAngle/numpy.float(valueNumber))%cv2.cv.CV_PI
            
#            x1 = numpy.uint16(-numpy.cos(accumAngle)*1000.0)
#            y1 = numpy.uint16(-numpy.sin(accumAngle)*1000.0)
#            x2= numpy.uint16(+numpy.cos(accumAngle)*1000.0)
#            y2= numpy.uint16(+numpy.sin(accumAngle)*1000.0)
            cv2.line(img, (self.xoffset, self.yoffset), (self.xoffset+numpy.int(accumAngleX), self.yoffset+numpy.int(accumAngleY)), cv2.cv.Scalar(255, 0, 0), 2, cv2.CV_AA, 0)    
#            cv2.line(img, (self.xoffset+numpy.int(x1), self.yoffset+numpy.int(y1)), (self.xoffset+numpy.int(x2), self.yoffset+numpy.int(y2)), cv2.cv.Scalar(255, 0, 0), 2, cv2.CV_AA, 0)
    #        cv2.line(img, (self.xoffset+numpy.int(numpy.median(lines[:, 0])), self.yoffset+numpy.int(numpy.median(lines[:, 1]))), (self.xoffset+numpy.int(numpy.median(lines[:, 2])), self.yoffset+numpy.int(numpy.median(lines[:, 3]))), cv2.cv.Scalar(255, 0, 0), 2, cv2.CV_AA, 0)
            self.angle = numpy.arctan(numpy.float(accumAngleY)/numpy.float(accumAngleX))
        else:
#            leng = numpy.sqrt(numpy.power(lines[3]-lines[1], 2)+numpy.power(lines[2]-lines[0], 2))
            accumAngle = self.angle + numpy.arctan(numpy.float(lines[3]-lines[1])/numpy.float(lines[2]-lines[0]))
            x2= numpy.uint16(numpy.cos(accumAngle)*100.0)
            y2= numpy.uint16(numpy.sin(accumAngle)*100.0)
            cv2.line(img, (self.xoffset, self.yoffset), (self.xoffset+numpy.int(x2), self.yoffset+numpy.int(y2)), cv2.cv.Scalar(255, 0, 0), 2, cv2.CV_AA, 0)
        return img