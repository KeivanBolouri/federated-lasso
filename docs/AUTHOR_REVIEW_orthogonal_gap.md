# AUTHOR_REVIEW: optional orthogonal objective-gap observation

Status: AI-proposed technical note for independent author review. This is not part of the manuscript, is not an author-approved new result, and is not a claim of originality. It is an elementary consequence of the Lasso KKT condition. The author should independently assess the mathematics and prior literature and write any proposed manuscript addition in their own words.

Under the manuscript's orthogonal assumptions, let theta = S(zbar, lambdabar), S = supp(theta), h = (zbar-theta)/lambdabar, and Delta(b) = F(b)-F(theta). Then

\[
\Delta(b)=\tfrac12\|b-\theta\|_2^2+
\bar\lambda\{\|b\|_1-\|\theta\|_1-h^\top(b-\theta)\}.
\]

Because h belongs to the l1 subgradient at theta, this implies

\[
\Delta(b)\ge\tfrac12\|b-\theta\|_2^2+
\sum_{k\notin S}(\bar\lambda-|\bar z_k|)|b_k|.
\]

If S complement is nonempty, define delta = min over k not in S of (lambdabar-|zbar_k|), which is nonnegative. For epsilon>0,

\[
\#\{k\notin S: |b_k|>\epsilon\}
\le \frac{\Delta(b)}{\epsilon\delta+\epsilon^2/2}.
\]

Proof outline: orthogonality gives F(b)=C+||b-zbar||²/2+lambdabar||b||1. Expand around theta and use theta-zbar=-lambdabar*h for the equality. On active coordinates, the remaining penalty contribution is lambdabar*(|b_k|-sign(theta_k)*b_k)>=0. On inactive coordinates it is lambdabar*|b_k|-zbar_k*b_k >= (lambdabar-|zbar_k|)*|b_k|. Each extra coefficient above epsilon contributes at least epsilon²/2+delta*epsilon to the lower bound. Sum those terms.

Interpretation to evaluate: small objective error can coexist with many tiny extra nonzero coefficients. Without a positive activity threshold or magnitude margin, a nontrivial exact-support guarantee does not follow from objective accuracy. The inequality concerns extra coordinates relative to the pooled optimizer, not false positives relative to the data-generating truth. It applies to any b, including the averaged fixed point, only under the orthogonal assumptions.

Validation: algebra checked by expansion and 10,000 randomized numerical examples spanning zero, nonzero and mixed pooled supports; identity and counting bound held at floating-point tolerance. This is a computational sanity check, not a substitute for author verification or a novelty assessment.
