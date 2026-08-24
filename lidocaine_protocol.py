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
"""

LIDOCAINE_PROTOCOL_TEXT = """\
2,6-Dimethylnitrobenzene (1.0 g) was dissolved in glacial acetic acid. A \
solution of SnCl2·2H2O (4.6 g) in concentrated HCl (8 mL) was then added, \
and the mixture was stirred and allowed to stand for 15 min, during which \
the reduced product crystallized as the amine hydrochloride salt. The \
mixture was cooled and the solid collected by Buchner filtration. The salt \
was basified with 30% KOH to liberate free 2,6-dimethylaniline, which was \
extracted with ether, washed, and dried over K2CO3, then evaporated to \
isolate the free aniline. (Commercially available 2,6-dimethylaniline may \
be substituted directly for this reduction step.)

The 2,6-dimethylaniline was then reacted with a slight excess of \
chloroacetyl chloride (7.2 g) in glacial acetic acid, warming the mixture \
to 40–50 °C on a steam bath. Aqueous sodium acetate (1 g) was added \
during workup to prevent HCl from protonating unreacted amine, which would \
otherwise co-precipitate with the product. The mixture was cooled and the \
resulting alpha-chloro-2,6-dimethylacetanilide was isolated by Buchner \
filtration and air-dried.

The chloroacetanilide intermediate was combined with toluene and a \
threefold molar excess of diethylamine and refluxed for approximately \
90 min, with the reaction monitored by TLC (chloroform eluent) as the \
chloride underwent SN2 displacement assisted by the adjacent carbonyl. The \
mixture was cooled to room temperature and then in an ice bath; the \
resulting crystals were filtered, and the product was purified by \
acid–base extraction (3 M HCl, then 30% KOH), extracted into pentane, \
dried over Na2CO3, and concentrated to isolate lidocaine free base.

The free base was dissolved in ether, and sulfuric acid in ethanol (2 mL \
of 2.2 M per gram of lidocaine) was added; crystallization was induced by \
scratching and the mixture was diluted with acetone to aid filtration. The \
precipitated lidocaine bisulfate was collected by Buchner filtration, \
rinsed with acetone, and air-dried.\
"""

LIDOCAINE_TARGET = "Lidocaine bisulfate"
LIDOCAINE_SYNTHESIS_TYPE = "multi-step organic synthesis (nitro reduction, acylation, SN2 alkylation, salt formation)"
LIDOCAINE_SOURCE = "Chemistry 212 lab manual, Cerritos College (not a journal article — no DOI)"
