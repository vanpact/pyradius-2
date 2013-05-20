"""
Created on Apr 11, 2013

@author: yvesremi
"""

import os
from sets import Set
import pandas
from PyQt4 import QtGui, QtCore

class ResultsWriter(object):
    """Class for outputing results in csv or xls files"""


    def __init__(self, fileName=None):
        """
        Constructor
        
        :param fileName: The name of the file where to output the results
        :type fileName: String   
        """
        self.parsedInfo = None
        self.columnsToPad = Set()
        self.fileName = None
        self.buffer = []
        self.fileExist = True
        fileExtension=None
        self.sheet=None
        if(fileName is not None):
            _, fileExtension = os.path.splitext(fileName)
        if(fileExtension =='.xlsx' or fileExtension =='.xls'):
            self.isExcel = True
            self.isCSV = False
        elif(fileExtension =='.csv' or fileExtension !='.txt'):
            self.isCSV = True
            self.isExcel = False
        elif(fileName!=None):
            val = ValueError('The input file must either be a csv, txt or an Excel(xls or xlsx) file.')
            val.args="inp"
            raise(val) 
        self.setWriteFile(fileName)
        
    def setWriteFile(self, fileName, sheet=None):
        """
        Read the file and put the results in a data frame
        
        :param fileName: The name of the file where to output the results
        :type fileName: String   
        :param sheet: argument only useful if the file format is xls or xlsx. It is the sheet name in the Excel file
        :type sheet: String 
        """
        if(isinstance(fileName, basestring)):
            self.fileName = unicode(fileName)
            if(self.isExcel):
                xlsFile=None
                try:
                    with open(fileName):
                        xlsFile = pandas.ExcelFile(self.fileName)
                except IOError:
                    self.fileExist = False
                self.preparexls(sheet, xlsFile)
            elif(self.isCSV):
                if(self.fileExist):
                    try:
                        with open(fileName):
                            self.parsedInfo = pandas.read_csv(self.fileName, sep='\t', header=0)
                    except IOError:
                        self.parsedInfo = pandas.DataFrame()
                        self.fileExist = False
        elif(fileName is None):
            self.fileExist = False
            self.parsedInfo = pandas.DataFrame()
            
                
    def preparexls(self, sheet=None, xlsFile=None):
        """
        Internal method, used when reading the file into a dataFrame.
  
        :param sheet: argument only useful if the file format is xls or xlsx. It is the sheet name in the Excel file
        :type sheet: String 
        :param xlsFile: The Excel file
        :type xlsFile: a pandas ExcelFile 
        """
        if(self.fileExist):
            self.sheet=sheet
            if(self.sheet==None):
                self.sheet = xlsFile.sheet_names[0]
            self.parsedInfo = xlsFile.parse(self.sheet, header=0)
        else:
            self.parsedInfo = pandas.DataFrame()
            self.sheet='sheet1'
        
    def addDatas(self, info):
        """
        Push data in the list of data to write to the file
        
        :param info: The information to add
        :type info: a dictionnary or a list of dictionnary
        """
        if(isinstance(info, list)):
            for inf in info:
                self.buffer.append(inf)
        else:
            self.buffer.append(inf)
        
    def mergeDatas(self, info):
        """
        Internal method, merge the info with the current dataFrame
        
        :param info: The information merge
        :type info: a list of dictionnaries
        """
        if(self.fileExist):
            if([col for col in self.parsedInfo.columns].count('Time')==1):
                self.parsedInfo = self.parsedInfo.set_index(keys=[str('Time')], inplace=False)
#         info = pandas.DataFrame(info).dropna(how = 'any', subset = ['Time'])
        dataToMerge = pandas.DataFrame(info)
        dataToMerge = dataToMerge.set_index(keys=[str('Time')], inplace=False)
        for colName in dataToMerge.columns:
            for colName1 in self.parsedInfo.columns:
                if(colName == colName1):
                    self.parsedInfo = self.parsedInfo.drop(colName1, axis=1)
            self.columnsToPad.add(colName)
        self.parsedInfo = pandas.merge(self.parsedInfo, dataToMerge, left_index=True, right_index=True, copy=False, how='outer')
        
    def write(self, padNewColumn=True, outFileName=None, sheet=None):
        """
        Write the info added with the method addData to the outputFile
        
        :param padNewColumn: the write file already has information and sampling of the file is higher, pad the new data to add so there is no holes in the modifiers/added columns
        :type padNewColumn: bool
        :param outFileName: The output file name
        :type outFileName: String
        :param sheet: argument only useful if the file format is xls or xlsx. It is the sheet name in the Excel file
        :type sheet: String 
        """
        self.mergeDatas(self.buffer)
        self.buffer = []
        if(padNewColumn):
            for colName in self.columnsToPad:
                    self.parsedInfo[colName].fillna(method='ffill', inplace=True)
            for colName in self.columnsToPad:
                    self.parsedInfo[colName].fillna(method='bfill', inplace=True)
        toCheck=set()    
        for colName in self.parsedInfo.columns:
            toCheck.add(colName)
        toCheck.difference(toCheck, self.columnsToPad)
        colNameList = [colName for colName in toCheck]
        self.parsedInfo = self.parsedInfo.dropna(how = 'all', subset = colNameList)
        if(outFileName is None):
            outFileName = self.fileName
        _, fileExtension = os.path.splitext(outFileName)
        if(sheet is None):
            if(self.sheet is None):
                self.sheet='sheet1'
            sheet=self.sheet
        if(fileExtension =='.xlsx' or fileExtension =='.xls'):
            self.parsedInfo.to_excel(outFileName, sheet, header=True, index=True, index_label='Time')
        elif(fileExtension =='.csv' or fileExtension !='.txt'):
            self.parsedInfo.to_csv(outFileName, sep='\t', na_rep='', float_format='%.6f', header=True, index=True, index_label='Time', mode='w')
        else:
            val = ValueError('The output file must either be a csv, txt or an Excel(xls or xlsx) file.')
            val.args="out"
            raise(val)

        