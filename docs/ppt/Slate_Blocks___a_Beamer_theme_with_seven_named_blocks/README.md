# slate-blocks

A 16:10 beamer deck built on stock themes — Madrid, miniframes, circles — with
three things added: a palette derived from one colour, seven named blocks in
place of beamer's three, and a photographic title page.

The name is the palette, not the season. `#bec2cb` is a desaturated blue-grey;
the autumn cover is a photograph you are expected to swap out, and swapping it
changes nothing about the colours.

```
./check.sh          build slides.tex, report overflows
./check.sh demo     build demo.tex — every construct, filled
```

pdfLaTeX or XeLaTeX; either works. Nothing here loads a font by filename, so
unlike `midcentury-olive/` there is no font directory and nothing to install.

## Before you build: the cover, and the logo

`figures/background.png` is the author's own photograph, edited into a
watercolour. It is used twice — full-bleed on the title page, and again at
6.7% as the watermark under every other slide. One line sets both:

```latex
\slateWatermark{figures/background.png}   % preamble, in slides.tex
\renewcommand{\slateBackground}{figures/background.png}   % if you want them different
```

Replace it and check two things. The title box is a pale gradient, so it
vanishes over a pale sky — look at the cover, not at the thumbnail. And 6.7%
was set against an image of this lightness; a darker photograph at the same
opacity will fight the body text.

`figures/logo.png` is a placeholder. No institution's mark ships here — a logo
is its owner's trademark and the licence on these files does not extend to it.
Replace it locally and keep it out of the history.

## The palette is one colour

```latex
\definecolor{ThemeColor}{RGB}{190, 194, 203}   % #bec2cb
```

Everything else is derived from it by colour-wheel relations — two analogous,
three tetradic, two split-complementary — and all seven sit at the same
lightness and saturation. The hue distinguishes them and nothing else does, so
no block shouts louder than another.

The same colour does both ends of the range: at full strength it is a
background, and at `!250` — beamer's syntax for a tint past 100%, that is, a
darkening — it is the body text. That is what keeps the deck quiet.

To re-tint the whole deck, change the one `\definecolor` and re-derive the
other seven. They are not computed at build time; they are written out, because
`xcolor`'s wheel arithmetic does not survive being read back.

## Seven blocks

Beamer gives you `block`, `exampleblock`, `alertblock`. This wanted seven, each
naming what it holds:

```latex
\newcolouredblock{ExampleBox}   {Example}    {ThemeColor}
\newcolouredblock{CommentBox}   {Comment}    {AnalogousColor-1}
\newcolouredblock{ProposalBox}  {Proposal}   {AnalogousColor-2}
\newcolouredblock{ReminderBox}  {Reminder}   {TetradicColor-1}
\newcolouredblock{ProblemBox}   {Problem}    {TetradicColor-2}
\newcolouredblock{ChallengeBox} {Challenge}  {TetradicColor-3}
\newcolouredblock{DefinitionBox}{Definition} {SplitComplementaryColor-1}
```

The second argument is the printed label and nothing depends on it — rename
them into your own language there.

`\newcolouredblock` exists because beamer has no per-block colour.
`\setbeamercolor` inside a frame leaks into everything after it, so the factory
sets the colours before the environment and puts them back after, using
etoolbox's `\BeforeBeginEnvironment` / `\AfterEndEnvironment`. The original deck
wrote those eight lines out once per block, seven times over.

## The SWOT grid

```latex
\swot{strengths}{weaknesses}{opportunities}{threats}
```

A 3×3 `tcbitemize`: a blank corner, two column headers, two rows each opening
with a rotated label. The four quadrant colours are **mixed, not picked** —
every quadrant is half helpful/harmful and half internal/external — so both
axes are legible in the colour itself and a reader can place a quadrant without
reading a label. The letter watermarked in each quadrant is the tcolorbox
colour name, which labels it without spending a line on a heading.

Keep the contents short. Each quadrant gets a quarter of the slide and about
four lines is what fits; `./check.sh` will tell you when it does not.

The same command is in `midcentury-olive/`, in that template's colours.

## These stop the build

**`\newtheorem` on a name that already exists.** `\Example`, `\Definition` and
several others are taken — amsmath and beamer's own theorem set define them —
and `\newtheorem` fails outright. That is why every block name here carries a
`Box` suffix. It is uglier than the bare word and cheaper than finding out
which words are free in every package a future deck might load.

**A `\newcommand` with an argument inside a frame.** Beamer reads a frame body
twice, and the second read finds the definition already made:
`Illegal parameter number in definition of \beamer@doifinframe`. Define it in
the preamble.

## These render wrong and say nothing

**One pass.** Anything positioned with `remember picture, overlay` — which is
the whole title page, and the watermark — needs the previous pass's `.aux`. A
single `xelatex` leaves the cover blank and reports nothing. `check.sh` always
builds twice; if you build by hand, do the same.

**A cover photograph that does not cover the page.** The deck is 16:10 and most
photographs are 16:9. `\includegraphics[width=\paperwidth]` scales the picture
to the paper's width and leaves it about a tenth of the page short — a white
band across the top that is easy to miss on a thumbnail. Giving both `width`
and `height` distorts the picture; adding `keepaspectratio` letterboxes it,
which is the same gap arrived at politely. `\slatecover` measures the image at
full width and scales by height instead when that is not tall enough, and the
node sits at `current page.center` so whichever dimension overruns is cropped
evenly by the page edge. Both the cover and the watermark go through it.

**A title box over a pale sky.** The box is `AnalogousColor-1` shaded from 75%
to 25% at `shading angle=60`, double-stroked. The double stroke and the
diagonal gradient are both there so that neither edge matches the photograph
behind it for long. Over a flat pale ground they are invisible and the box
looks like a bug.

**The watermark at the wrong strength.** 6.7% reads as paper texture. At 15% it
reads as an image, and the body text has to fight it. It is one number in
`\slateWatermark`, and it is worth re-checking after you change the photograph.

**Cover opacity.** The photograph is laid down at `opacity=0.55`. At full
strength it and the title box sit at the same weight and the box stops reading
as something over it; below about 0.5 the picture stops looking held back and
starts looking faded.

The usual recipe is two layers — the image at 0.65 with a white rectangle at
0.15 over it. It is not worth the second layer here. The page behind is white,
so the veil does nothing the opacity cannot, and the two render to within one
part in 255 of a single `opacity=0.5525`. That was measured by differencing the
two PNGs, not assumed. A white veil only earns its place over a ground that is
not already white.

## Noise in the log

Titled frames on this theme report an `Overfull \hbox` of a few points that
does not respond to the width of anything on the page — it was traced by
building the SWOT page at four different grid widths and getting the same
`6.03252pt` each time. `check.sh` reports it rather than hiding it; if a number
does not move when you change the thing it names, it is not measuring that
thing. The report that matters is the first section, *frames whose content is
too tall*, and that one is exact.

## What is in `style.tex`

| | |
|---|---|
| the palette | one `\definecolor` and seven derivations, then the beamer colour assignments |
| `\newcolouredblock` | the block factory, and the seven it builds |
| `\slatecover{file}` | scales an image to cover the slide at any aspect ratio, without distorting it |
| `\slateWatermark` | the cover photograph on every slide at 6.7% |
| `\slatetitlepage` | full-bleed photograph, double-stroked gradient title box, author/affiliation/date/logo |
| `\swot` | the 3×3 grid, its axis labels and its mixed quadrant colours |

Load it after `\documentclass` and the `\usetheme` lines. The five title-page
fields are `\renewcommand`s, not arguments, because a title page is the one
place a deck wants long strings with markup in them and five arguments to one
command is unreadable.
