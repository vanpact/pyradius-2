#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on Oct 15, 2012

@author: yvesremi
'''

from pycana import CodeAnalyzer
import View, PreTreatments, imageConverter, VideoWidget
if __name__ == '__main__':
    analyzer = CodeAnalyzer(View)
    relations = analyzer.analyze()
    analyzer.draw_relations(relations, 'class_diagram.png')