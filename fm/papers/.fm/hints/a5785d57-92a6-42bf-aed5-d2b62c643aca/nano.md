Educational infographic, clean flat design, 16:9 landscape, white background, soft blue and orange accent palette, thin dark-gray outlines, crisp sans-serif labels, generous whitespace, no photorealism.

Title at the top center in bold: "Sequence Packing (시퀀스 패킹)". Small subtitle under it: "One forward pass, many crop sizes".

Layout: three panels arranged left to right, connected by large right-pointing arrows so the eye flows from problem to trick to guarantee.

Panel 1 (left), header "Problem: different lengths". Show two token rows made of small square tiles. The top row is long: about 16 blue tiles, labeled "Global crop 224 -> 257 tokens". The bottom row is short: about 4 orange tiles, labeled "Local crop 98 -> 50 tokens". Between the rows a red "not equal" symbol and a small red X over a dashed box that tries to enclose both, labeled "Cannot batch together". Below the rows a small caption: "Old way: separate forwards" with two tiny ViT block icons side by side.

Panel 2 (center), header "Trick: concatenate". Show the blue tiles and orange tiles laid end to end in a single long horizontal row: blue segment, then orange segment, then another short orange segment. A bracket under the whole row labeled "One long sequence". Above the row a single ViT block icon labeled "Transformer blocks as usual". A small tag in the corner: "From NLP (Krell 2022)".

Panel 3 (right), header "Guard: block-diagonal mask". Draw a square attention matrix grid. Along the diagonal draw filled squares of different sizes: one large blue square in the top left, then two smaller orange squares along the diagonal below it. All off-diagonal regions are light gray with faint hatching, labeled "masked, no cross-attention". Diagonal blocks labeled "attend within own crop only". Under the matrix a green check mark with text "Equivalent to separate forwards".

Bottom band across the whole image, light blue background, with three short bullet chips: "Fewer kernel launches", "Higher GPU utilization", "Same result, faster training". Bottom right small text: "DINOv2, xFormers BlockDiagonalMask".

Keep all labels short and legible, consistent icon style, aligned grid, high contrast text.
