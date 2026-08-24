"""
Lidocaine synthesis — raw protocol text (Part 1 demo input)
=============================================================
Free-text procedure, not a pre-extracted operations list. This is the input
to pipeline.parse_protocol_text() / run_round1_from_text() — the LLM reads
this prose directly and identifies the operations, materials, quantities,
and conditions itself, rather than being handed them pre-extracted.

Source: *Synthesis of Lidocaine*, Chemistry 212 Laboratory manual, Cerritos
College. https://www.cerritos.edu/chemistry/chem_212/Documents/Lab/10_lidocaine.pdf
Standard four-step route (isolated here as the bisulfate salt, which is
easier to purify than the hydrochloride used clinically). Not a journal
article, so no DOI exists for it.

This text was checked directly against the source PDF (fetched and read in
full) rather than transcribed from an earlier paraphrase in this project.
That check caught two things worth noting: several quantities were missing
from an earlier draft (now restored below), and an earlier draft included a
mechanistic "why" for the sodium acetate addition in step B (that it
prevents HCl from protonating unreacted amine) which does NOT appear
anywhere in the source document -- the manual states the step with no
rationale given. That sentence has been removed here rather than carried
forward as if it were part of the source protocol.
"""

LIDOCAINE_PROTOCOL_TEXT = """\
2,6-Dimethylnitrobenzene (1.0 g) was dissolved in 10 mL glacial acetic \
acid. A separately prepared solution of SnCl2·2H2O (4.6 g) in 8 mL \
concentrated HCl was added in one portion; the mixture was swirled and \
left to stand for 15 min, then cooled, and the crystalline amine \
hydrochloride salt was collected by Buchner filtration. The moist salt was \
redissolved in 5–10 mL water and made strongly basic with 30% KOH \
(12–17 mL) to liberate free 2,6-dimethylaniline, which was extracted with \
three 10 mL portions of ether, rinsed twice with 10 mL water, dried over \
K2CO3, and evaporated to isolate the free aniline. (Commercially available \
2,6-dimethylaniline may be substituted directly for this reduction step, \
since it costs less per gram than the nitro compound.)

For every 7 g of 2,6-dimethylaniline, 50 mL glacial acetic acid and 7.2 g \
chloroacetyl chloride were added in that order, and the solution was \
warmed on a steam bath to 40–50 °C. After removing from the bath, a \
solution of 1 g sodium acetate in 100 mL water was added. The mixture was \
cooled and the resulting alpha-chloro-2,6-dimethylacetanilide was \
collected by Buchner filtration and air-dried.

The chloroacetanilide intermediate, in 25 mL toluene, was combined with a \
threefold molar excess of diethylamine, fitted with a reflux condenser, \
and refluxed vigorously, with the reaction monitored by TLC (chloroform \
eluent) at 15–30 min intervals. After the starting material was consumed \
or after 90 min of refluxing, whichever came first, the mixture was cooled \
to room temperature and then in an ice bath; the resulting crystals were \
filtered and rinsed with a small amount of pentane. The filtrate was \
extracted with two 10 mL portions of 3 M HCl, the acidic aqueous layer was \
made strongly basic with 30% KOH, extracted with two 10 mL portions of \
pentane, rinsed with six 10 mL portions of water to remove unreacted \
diethylamine, dried over Na2CO3, and concentrated, combining with the \
filtered crystals to give lidocaine free base.

The lidocaine free base was dissolved in ether (10 mL per gram) and 2 mL \
of 2.2 M sulfuric acid in ethanol per gram of lidocaine was added. The \
mixture was stirred and scratched with a glass rod to induce \
crystallization, then diluted with an equal volume of acetone to aid \
filtration. The precipitated lidocaine bisulfate was collected by Buchner \
filtration, rinsed with acetone, and air-dried.\
"""

LIDOCAINE_TARGET = "Lidocaine bisulfate"
LIDOCAINE_SYNTHESIS_TYPE = "multi-step organic synthesis (nitro reduction, acylation, SN2 alkylation, salt formation)"
LIDOCAINE_SOURCE = "Chemistry 212 lab manual, Cerritos College (not a journal article — no DOI)"
