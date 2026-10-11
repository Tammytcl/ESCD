"""CPU numerical contracts; these are not end-to-end reproduction tests."""
import math
from types import SimpleNamespace
import unittest

import torch

from opd.cross_tokenizer.bpm_core import BPMRoutingArtifact, compile_longest_prefix_routes, RESIDUAL_ROUTE, build_bpm_rows
from opd.cross_tokenizer.bpm_dense import compile_dense_route_plan, project_dense_teacher_rows, bpm_projected_divergence_from_log_probs
from opd.cross_tokenizer.escd_core import StudentBytePrefixIndex, compile_child_events, compile_full_vocabulary_plan
from opd.cross_tokenizer.escd import ESCDCrossTokenizer
from opd.cross_tokenizer.gold import GOLDCrossTokenizer
from opd.cross_tokenizer.simct import SimCTCrossTokenizer, SimCTOfficialCrossTokenizer
from opd.cross_tokenizer._paper_utils import merge_observed_probabilities
from opd.cross_tokenizer.xtoken_core import build_projection_matrix


class Distribution:
    def __init__(self, logits):
        self.logits = logits
        self.length = len(logits)

    def rows(self, indices):
        return Distribution(self.logits[indices])

    def selected_log_probs(self, ids, *, vocab_size, temperature=1.):
        assert vocab_size == self.logits.shape[-1]
        return (self.logits / temperature).log_softmax(-1).gather(1, ids)

    def selected_logits(self, ids):
        return self.logits.gather(1,ids)


class ToyTokenizer:
    all_special_ids = []

    def __init__(self, pieces, encodings=None):
        self.pieces = pieces
        self.encodings = encodings or {}

    def get_vocab(self):
        return {p:i for i,p in enumerate(self.pieces)}

    def decode(self, ids, **kwargs):
        return ''.join(self.pieces[i] for i in ids)

    def encode(self, text, **kwargs):
        if text in self.encodings:
            return self.encodings[text]
        return [self.pieces.index(text)]


class ObjectiveContracts(unittest.TestCase):
    def setUp(self):
        self.teacher = (b'Bread', b'Breads', b'Break', b'Br', b'Z')
        self.student = (b'Br', b'ead', b'eads', b'eak', b'eaks', b'x')
        routes = compile_longest_prefix_routes(self.teacher, self.student)
        self.artifact = BPMRoutingArtifact(routes, self.teacher, self.student, (), (), tuple(range(6)), 'toy-contract')
        self.index = StudentBytePrefixIndex(self.student)

    def test_bpm_mass_and_residual(self):
        self.assertEqual(self.artifact.teacher_to_student, (0,0,0,0,RESIDUAL_ROUTE))
        plan = compile_dense_route_plan(self.artifact.teacher_to_student, device=torch.device('cpu'))
        mapped, residual = project_dense_teacher_rows(torch.tensor([[.2,.1,.15,.25,.3]]),plan)
        torch.testing.assert_close(mapped,torch.tensor([[.7]]))
        torch.testing.assert_close(residual,torch.tensor([.3]))

    def test_bpm_fkl_loss_and_gradient(self):
        z=torch.tensor([[.2,-.3]],dtype=torch.float64,requires_grad=True)
        q=torch.tensor([[.7,.3]],dtype=torch.float64)
        loss=bpm_projected_divergence_from_log_probs(q,z.log_softmax(-1),beta=0.,skew=.1).sum()
        expected=.7*math.log(.7)+.3*math.log(.3)-(.7*.2+.3*(-.3))+math.log(math.exp(.2)+math.exp(-.3))
        self.assertAlmostEqual(loss.item(),expected,places=12)
        loss.backward();torch.testing.assert_close(z.grad,z.detach().softmax(-1)-q)

    def test_bpm_extreme_log_probability_keeps_gradient(self):
        z=torch.tensor([[-1000.,0.]],requires_grad=True)
        loss=bpm_projected_divergence_from_log_probs(torch.tensor([[1.,0.]]),z.log_softmax(-1),beta=0.,skew=.1).sum()
        self.assertEqual(loss.item(),1000.);loss.backward()
        torch.testing.assert_close(z.grad,torch.tensor([[-1.,1.]]))

    def test_bpm_truncated_response_does_not_invent_stop(self):
        rows,_,_=build_bpm_rows(student_response_ids=[0,1],teacher_input_ids=[0,0],assistant_spans=[dict(student_start=0,student_end=2,teacher_start=1,teacher_end=2)],artifact=self.artifact)
        self.assertEqual([x['kind'] for x in rows],['aligned','interior'])
        self.assertEqual(rows[1]['prefix_hex'],b'Br'.hex())

    def test_bpm_terminal_stop_correspondence(self):
        a=BPMRoutingArtifact(self.artifact.teacher_to_student+(RESIDUAL_ROUTE,),self.teacher+(None,),self.student+(None,),(5,),(6,),tuple(range(7)),'toy-stops')
        rows,_,_=build_bpm_rows(student_response_ids=[0,1,6],teacher_input_ids=[0,0,5],assistant_spans=[dict(student_start=0,student_end=3,teacher_start=1,teacher_end=3)],artifact=a)
        self.assertEqual([x['kind'] for x in rows],['aligned','interior','stop'])
        self.assertEqual(rows[0]['student_stop_token_id'],6)
        with self.assertRaisesRegex(ValueError,'stop-token structure differs'):
            build_bpm_rows(student_response_ids=[0,1],teacher_input_ids=[0,0,5],assistant_spans=[dict(student_start=0,student_end=2,teacher_start=1,teacher_end=3)],artifact=a)

    def test_escd_completion_groups_and_mass(self):
        c=compile_child_events(teacher_token_ids=range(5),teacher_probabilities=[.2,.1,.15,.25,.3],actual_first_token_id=0,artifact=self.artifact,student_index=self.index)
        events={e.remaining_bytes:e for e in c.events}
        self.assertEqual(set(events),{b'ead',b'eak'})
        self.assertEqual(set(events[b'ead'].success_token_ids),{1,2})
        self.assertAlmostEqual(events[b'ead'].mass,.3)
        self.assertAlmostEqual(events[b'eak'].mass,.15)
        self.assertAlmostEqual(c.supported_mass,.45)
        plan=compile_full_vocabulary_plan(artifact=self.artifact,student_index=self.index)
        self.assertEqual({plan.component_targets[i] for i in plan.components_after(b'Br')},{b'Bread',b'Break'})

    def test_escd_actual_child_scorer_and_gradient(self):
        c=compile_child_events(teacher_token_ids=range(5),teacher_probabilities=[.2,.1,.15,.25,.3],actual_first_token_id=0,artifact=self.artifact,student_index=self.index)
        z=torch.tensor([[.1,.4,-.3,.2,.5,-.2]],requires_grad=True)
        ctx=SimpleNamespace(student=SimpleNamespace(response_token_ids=torch.tensor([0]),distribution=Distribution(z)))
        plugin=object.__new__(ESCDCrossTokenizer);plugin._bpm_artifact=self.artifact
        actual=plugin._score_child_targets(ctx,[(0,c.events)],student_row_chunk_size=1).sum()
        p=z.softmax(-1);expected=-.3*(p[0,1]+p[0,2]).log()-.15*(p[0,3]+p[0,4]).log()
        torch.testing.assert_close(actual,expected)
        ga=torch.autograd.grad(actual,z,retain_graph=True)[0];ge=torch.autograd.grad(expected,z)[0]
        torch.testing.assert_close(ga,ge)

    def test_simct_rkl_masks_and_teacher_detach(self):
        s=torch.tensor([[.2,.8,4.]],requires_grad=True);t=torch.tensor([[.9,.1,9.]],requires_grad=True)
        mask=torch.tensor([[True,True,False]])
        actual=SimCTCrossTokenizer._reverse_kl(s,t,mask).sum()
        p=s[:,:2].softmax(-1);q=t.detach()[:,:2].softmax(-1)
        expected=(p*(p.log()-q.log())).sum();torch.testing.assert_close(actual,expected)
        actual.backward();self.assertIsNone(t.grad);self.assertEqual(s.grad[0,2].item(),0.)

    def test_simct_paper_and_official_score_spaces_are_distinct(self):
        z=torch.tensor([[1.,2.,3.],[5.,1.,0.]])
        shifted=z+torch.tensor([[10.],[-7.]])
        ids=torch.tensor([[1],[0]])
        paper=object.__new__(SimCTCrossTokenizer);official=object.__new__(SimCTOfficialCrossTokenizer)
        first=paper._selected_scores(Distribution(z),ids,vocab_size=3)
        second=paper._selected_scores(Distribution(shifted),ids,vocab_size=3)
        torch.testing.assert_close(first,second)
        raw=official._selected_scores(Distribution(z),ids,vocab_size=3)
        self.assertFalse(torch.allclose(raw,first))

    def test_gold_observed_merge_is_not_renormalized(self):
        p=torch.tensor([[.2,.8],[.6,.4],[.3,.7]])
        got=merge_observed_probabilities(p,torch.tensor([0,2]),torch.tensor([2,3]),torch.tensor([1,0,1]))
        torch.testing.assert_close(got,torch.tensor([[.12,.48],[.3,.7]]))

    def test_gold_hybrid_components_and_reduction(self):
        s=torch.tensor([[.2,.3,.5]],requires_grad=True);t=torch.tensor([[.4,.35,.15,.1]])
        sm,su=GOLDCrossTokenizer._split_hybrid_probabilities(s,torch.tensor([0]),padded_unmatched_size=3)
        tm,tu=GOLDCrossTokenizer._split_hybrid_probabilities(t,torch.tensor([0]),padded_unmatched_size=3)
        got=GOLDCrossTokenizer._generalized_jsd_rows(sm,tm,beta=0.)+(su-tu).abs().sum(-1)
        self.assertAlmostEqual(got.item(),.4*math.log(2)+.4,places=6)
        # Two samples with different group counts must have equal total weight.
        total=100;scales=[GOLDCrossTokenizer._hybrid_normalization_scale(student_active_tokens=n,aligned_groups=g,step_active_token_count=total,step_sample_count=2) for n,g in [(20,2),(80,4)]]
        self.assertEqual([scales[0]*2/total,scales[1]*4/total],[.5,.5])

    def test_xtoken_fixed_projection_matches_rule(self):
        student=ToyTokenizer(['a','ab']);teacher=ToyTokenizer(['a','b'],{'ab':[0,1]})
        w=build_projection_matrix(student,teacher)
        self.assertEqual(w.row(0),((0,1.),))
        self.assertAlmostEqual(w.dense_row(1)[0],10/11)
        self.assertAlmostEqual(w.dense_row(1)[1],1/11)
        self.assertEqual(w.top1(1),0)


if __name__ == '__main__':
    unittest.main()
