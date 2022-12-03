import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler/passes")
from iterator_v3 import IterVarHyperGraph
from copy import deepcopy
import unittest


class IterVarHyperGraphTest(unittest.TestCase):
    def mm_default_1(self):
        iter_graph = IterVarHyperGraph(shape=[16384, 1024], names=['m', 'n'])
        iter_graph.view(shape=[512, 32, 1024])

        # get primals_5 iterators
        primals_5_graph = deepcopy(iter_graph)
        primals_5_graph.debroadcast(shape=[1024,])

        self.assertTrue(primals_5_graph.get_tensor_type() == "row")

        iter_graph.view(shape=[512, 512, 64])
    
    def bmm_default(self):
        iter_graph = IterVarHyperGraph(shape=[512, 512, 512], names=['b', 'm', 'n'])
        iter_graph.view(shape=(32, 16, 512, 512))

        mul_graph = deepcopy(iter_graph)
        mul_graph.debroadcast(shape=[32, 1, 1, 512])
        mul_graph.squeeze(2)
        mul_graph.squeeze(1)
        type = mul_graph.get_tensor_type()
        self.assertTrue(type == "row")
    
    def softmax_default(self):
        iter_graph = IterVarHyperGraph(shape=[262144, 512], names=['m', 'n'])
        iter_graph.view(shape=[32, 16, 512, 512])
        iter_graph.view(shape=[512, 512, 512])
    
    def bmm_default_1(self):
        iter_graph = IterVarHyperGraph(shape=[512, 512, 64], names=['b', 'm', 'n'])
        iter_graph.print_iter_vars()
        iter_graph.permute(permute_idx=[1, 0, 2])
        iter_graph.view(shape=[512, 32, 1024])
        iter_graph.view(shape=[16384, 1024])
        iter_graph.print_iter_vars()
    
    def test_bmm_default_6(self):
        iter_graph = IterVarHyperGraph(shape=[512, 64, 512], names=['b', 'm', 'n'])
        iter_graph.permute([2, 0, 1])
        iter_graph.view([512, 32, 1024])
        iter_graph.print_iter_vars()

if __name__ == '__main__':
    unittest.main()