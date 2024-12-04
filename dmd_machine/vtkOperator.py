import math

import numpy
import vtk
from vtk.numpy_interface import dataset_adapter as dsa


# All returned arrays are cast into either numpy or numarray arrays
arr = numpy.array


class vtkoperator:
    """Unstructured grid object to deal with VTK unstructured grids."""

    def __init__(self, filename=None):
        """Creates a vtu object by reading the specified file."""
        if filename is None:
            self.ugrid = vtk.vtkUnstructuredGrid()
        else:
            self.gridreader = None
            if filename[-4:] == ".vtk":
                self.gridreader = vtk.vtkGenericDataObjectReader()
                #print(vtk.vtkVersion.GetVTKMajorVersion())
            elif filename[-5:] == ".pvtu":
                self.gridreader = vtk.vtkXMLPUnstructuredGridReader()
            else:
                raise Exception("ERROR: don't recognise file extension" + filename)
            self.gridreader.SetFileName(filename)
            self.gridreader.Update()
            self.ugrid = self.gridreader.GetOutput()            
            #self.ugrid = dsa.WrapDataObject(self.ugrid)
            #if self.ugrid.GetNumberOfPoints() + self.ugrid.GetNumberOfCells() == 0:
                #raise Exception(
                    #"ERROR: No points or cells found after loading vtk " + filename
                #)
        self.filename = filename
    def AddScalarField(self, name, array):
        """Adds a scalar field with the specified name using the values from the
        array."""
        data = vtk.vtkDoubleArray()
        data.SetNumberOfValues(len(array))
        data.SetName(name)
        for i in range(len(array)):
            data.SetValue(i, array[i])

        if len(array) == self.ugrid.GetNumberOfPoints():
            pointdata = self.ugrid.GetPointData()
            pointdata.AddArray(data)
            pointdata.SetActiveScalars(name)
        elif len(array) == self.ugrid.GetNumberOfCells():
            celldata = self.ugrid.GetCellData()
            celldata.AddArray(data)
            celldata.SetActiveScalars(name)
        else:
            raise Exception("Length neither number of nodes nor number of cells")

    def AddVectorField(self, name, array):
        """Adds a vector field with the specified name using the values from the
        array."""
        n = array.size
        data = vtk.vtkDoubleArray()
        data.SetNumberOfComponents(array.shape[1])
        data.SetNumberOfValues(n)
        data.SetName(name)
        for i in range(n):
            data.SetValue(i, array.reshape(n)[i])

        if array.shape[0] == self.ugrid.GetNumberOfPoints():
            #print(self.ugrid.GetNumberOfPoints())
            pointdata = self.ugrid.GetPointData()
            pointdata.AddArray(data)
            pointdata.SetActiveVectors(name)
        elif array.shape[0] == self.ugrid.GetNumberOfCells():
            celldata = self.ugrid.GetCellData()
            celldata.AddArray(data)
        else:
            raise Exception("Length neither number of nodes nor number of cells")
         
    def GetFieldNames(self):
        """Returns the names of the available fields."""
        vtkdata = self.ugrid.GetPointData()
        return [vtkdata.GetArrayName(i) for i in range(vtkdata.GetNumberOfArrays())]
    
    def Write(self, filename=[]):
        """Writes the grid to a vtu file.
        If no filename is specified it will use the name of the file originally
        read in, thus overwriting it!
        """
        if filename == []:
            filename = self.filename
        if filename is None:
            raise Exception("No file supplied")
        if filename.endswith("pvtu"):
            gridwriter = vtk.vtkXMLPUnstructuredGridWriter()
        else:
            #gridwriter = vtk.vtkGenericDataObjectWriter()
            gridwriter = vtk.vtkUnstructuredGridWriter()
            #print("using vtk writer")
        gridwriter.SetFileName(filename)      
        #gridwriter.SetInputData(self.ugrid.VTKObject)
        if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
            gridwriter.SetInput(self.ugrid)
        else:
            gridwriter.SetInputData(self.ugrid)

        gridwriter.Write()