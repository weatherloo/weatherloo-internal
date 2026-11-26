"""Author-module parity, physical station losses, and time/query invariants."""

import unittest

import torch

from model import HyperDS, grid_average, subgrid_queries
from train import physical_station, station_loss


class FullHyperDSTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fusion_matches_author_forward(self):
        torch.manual_seed(3)
        model = HyperDS("paper",satellite_shape=(32,64)).eval()
        field,sat = torch.randn(2,5,8,16),torch.randn(2,2,4,32,64)
        with torch.no_grad():
            expected = model.fusion(sat,field)["block4"]
            actual = model.encode_field(field,model.encode_satellite(sat))
        torch.testing.assert_close(actual,expected)

    def test_satellite_changes_prediction_and_all_heads_have_gradients(self):
        torch.manual_seed(4)
        model = HyperDS("direct",satellite_shape=(32,64)).eval()
        field,sat,query = torch.randn(2,5,8,16),torch.randn(2,2,4,32,64),torch.rand(2,3,2)*2-1
        initial = model(field,sat,query,1)
        changed = model(field,sat+2,query,1)
        self.assertFalse(torch.allclose(initial,changed))
        initial.square().mean().backward()
        self.assertGreater(model.fusion.h8_conv3.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.fusion.pred_conv3.weight.grad.abs().sum().item(),0)
        for decoder in model.decoders:
            self.assertGreater(decoder.feat_fc1.weight.grad.abs().sum().item(),0)

    def test_time_extension_preserves_paper_at_hour_zero(self):
        torch.manual_seed(5)
        paper = HyperDS("paper",satellite_shape=(32,64)).eval()
        temporal = HyperDS("direct",satellite_shape=(32,64)).eval()
        temporal.load_state_dict(paper.state_dict())
        field,sat,query = torch.randn(2,5,8,16),torch.randn(2,2,4,32,64),torch.rand(2,3,2)*2-1
        with torch.no_grad():
            original = paper(field,sat,query)
            zero = temporal(field,sat,query,0)
            later = temporal(field,sat,query,24)
            reversed_queries = temporal(field,sat,query.flip(1),0).flip(1)
        torch.testing.assert_close(original,zero)
        torch.testing.assert_close(zero,reversed_queries)
        self.assertFalse(torch.allclose(zero,later))

    def test_subgrid_sampling_and_cell_average(self):
        stations = torch.tensor([[[-.99,-.99],[-.95,-.95],[.4,.8]]])
        query = subgrid_queries(1,4,8,"cpu",stations=stations)
        self.assertEqual(query.shape,(1,4*8*10,2))
        self.assertTrue(((query>=-1)&(query<=1)).all())
        for station in stations[0]:
            self.assertTrue((query[0]==station).all(dim=-1).any())
        averaged = grid_average(torch.full((1,4*8*10,5),2.),4,8)
        torch.testing.assert_close(averaged,torch.full((1,5,4,8),2.))

    def test_wind_loss_uses_physical_components_and_masks_missing_labels(self):
        prediction = torch.tensor([[[3.,4.,10.,900.,1.]]],requires_grad=True)
        target = torch.tensor([[[5.,10.,900.,float("nan")]]])
        actual = physical_station(prediction,torch.zeros(5),torch.ones(5))
        torch.testing.assert_close(actual,torch.tensor([[[5.,10.,900.,1.]]]))
        loss = station_loss(prediction,target,torch.zeros(5),torch.ones(5),[0])
        self.assertEqual(loss.item(),0)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_autoregressive_field_feedback_has_a_gradient_to_earlier_prediction(self):
        torch.manual_seed(6)
        model = HyperDS("autoregressive",satellite_shape=(32,64)).eval()
        initial = torch.randn(2,5,4,8)
        satellite = torch.randn(2,2,4,32,64)
        sat = model.encode_satellite(satellite)
        first_context = model.encode_field(initial,sat)
        feedback_query = subgrid_queries(2,4,8,"cpu",random=False)
        first = model.decode(first_context,initial,feedback_query,0)
        first.retain_grad()
        next_field = grid_average(first,4,8)
        context = model.encode_field(next_field,sat)
        second = model.decode(context,next_field,torch.zeros(2,1,2),1)
        second.square().mean().backward()
        self.assertIsNotNone(first.grad)
        self.assertGreater(first.grad.abs().sum().item(),0)


if __name__=="__main__":
    unittest.main()
