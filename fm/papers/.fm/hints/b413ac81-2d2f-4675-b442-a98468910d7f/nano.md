Educational infographic, clean flat design, 16:9 landscape, white background, muted navy / teal / coral accent colors, thin sans-serif labels.

Title at top center, bold: "DINOv2 Training Objective = DINO + iBOT + SwAV" with a small Korean subtitle "DINOv2 학습 목적함수".

Layout: three main panels side by side in the middle row, plus two smaller add-on boxes in a bottom row connected upward with dashed arrows labelled "+ add-on".

Panel 1 (left), header "DINO loss (image-level)": a photo icon of a dog shown as two different crops, one going into a box labelled "Student ViT", the other into a box labelled "Teacher ViT (EMA)". Both output a single highlighted token labelled "class token". Arrow from teacher to student with a label "cross-entropy". Small caption "global meaning" and Korean "이미지 전체".

Panel 2 (center), header "iBOT loss (patch-level)": same dog image drawn as a grid of patches; on the student side several patches are grey with a "?" labelled "masked patches", on the teacher side the full grid is visible. Arrow connecting a masked student patch to the matching teacher patch, label "predict masked patch". Small caption "local meaning" and Korean "패치 단위".

Panel 3 (right), header "SwAV centering (teacher)": a small bar chart of prototype scores where one bar is very tall labelled "collapse", an arrow labelled "Sinkhorn-Knopp x3" pointing to a second bar chart with even bars labelled "balanced". Small caption "prevents collapse".

Bottom row, add-on box A, header "KoLeo regularizer": left mini-diagram shows dots clumped together on a circle labelled "clumped", arrow labelled "push apart" to a circle with dots evenly spread labelled "uniform spread", small tag "weight 0.1" and Korean "특징 퍼뜨리기".

Bottom row, add-on box B, header "Short high-res phase": a horizontal training timeline bar, most of it labelled "224 px pretraining", the last short segment highlighted coral labelled "518 px, last 10k iters", small tag "cheap, near full high-res quality".

Visual flow: eyes read title, then left to right across the three loss panels joined by big "+" signs, then down to the two add-on boxes. Use consistent icons, generous whitespace, no long sentences, all labels under five words.
