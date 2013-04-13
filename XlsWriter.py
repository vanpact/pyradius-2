'''
Created on Apr 11, 2013

@author: yvesremi
'''
import os
from sets import Set
import pandas

class XlsWriter(object):
    '''
    classdocs
    '''


    def __init__(self, fileName=None):
        '''
        Constructor
        '''
        self.parsedInfo = None
        self.columnsToPad = Set()
        self.fileName = None
        self.buffer = []
        if(fileName is not None):
            self.setWriteFile(fileName)
        
    def setWriteFile(self, fileName, sheet=None):
        self.writeFile = None
        if(isinstance(fileName, basestring)):
            _, fileExtension = os.path.splitext(fileName)
            if(fileExtension == '.xls' or fileExtension == '.xlsx'):
                self.fileName = unicode(fileName)
                self.writeFile = pandas.ExcelFile(self.fileName)
                self.prepare(sheet)
                
    def prepare(self, sheet=None):
        self.sheet=sheet
        if(self.sheet==None):
            self.sheet = self.writeFile.sheet_names[0]
        self.parsedInfo = self.writeFile.parse(self.sheet, index_col=0)
        
    def addDatas(self, info):
        self.buffer.append(info)
        
    def mergeDatas(self, info):
        dataToMerge = pandas.DataFrame(info).set_index('Time')
        for colName in dataToMerge.columns:
            self.columnsToPad.add(colName)
        self.parsedInfo = pandas.merge(self.parsedInfo, dataToMerge, left_index=True, right_index=True, how='outer')
    
    def write(self, padNewColumn=True):
        self.mergeDatas(self.buffer)
        self.buffer = []
        if(padNewColumn):
            for colName in self.columnsToPad:
                    self.parsedInfo[colName].fillna(method='ffill', inplace=True)
            for colName in self.columnsToPad:
                    self.parsedInfo[colName].fillna(method='bfill', inplace=True)
            self.parsedInfo.dropna()
        self.parsedInfo.to_excel(self.fileName, self.sheet, header=True, index=True, index_label='Time')
        