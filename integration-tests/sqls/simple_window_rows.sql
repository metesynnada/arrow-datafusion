SELECT
    SUM(c2) OVER(ROWS BETWEEN 3 PRECEDING AND 1 FOLLOWING) as summation1,
    SUM(c5) OVER(ORDER BY c4 ROWS BETWEEN 3 PRECEDING AND 1 FOLLOWING) as summation2,
    SUM(c6) OVER(ORDER BY c3 ROWS 3 PRECEDING) as summation3,
    SUM(c3) OVER(ROWS BETWEEN 3 PRECEDING AND UNBOUNDED FOLLOWING) as summation4,
    SUM(c6) OVER(ROWS BETWEEN UNBOUNDED PRECEDING AND 1 FOLLOWING) as summatio5,
    SUM(c6) OVER(ORDER BY c13 ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as summation6,
    SUM(c6) OVER(ORDER BY c13 ROWS UNBOUNDED PRECEDING) as summation7,
    SUM(c6) OVER(PARTITION BY c1 ROWS BETWEEN 3 PRECEDING AND UNBOUNDED FOLLOWING) as summation8,
    SUM(c6) OVER(PARTITION BY c2 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 FOLLOWING) as summation9,
    SUM(c2) OVER(PARTITION BY c5 ORDER BY c13 ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as summation10,
    SUM(c6) OVER(PARTITION BY c1 ORDER BY c13 ROWS UNBOUNDED PRECEDING) as summation11,
    SUM(c4) OVER(PARTITION BY c6 ROWS BETWEEN 3 PRECEDING AND UNBOUNDED FOLLOWING) as summation12,
    SUM(c3) OVER(PARTITION BY c1 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 FOLLOWING) as summation13,
    SUM(c2) OVER(PARTITION BY c1 ORDER BY c13 ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as summation14,
    SUM(c6) OVER(PARTITION BY c13 ORDER BY c13 ROWS UNBOUNDED PRECEDING) as summation15,
    SUM(c3) OVER(PARTITION BY c1 ROWS BETWEEN 3 PRECEDING AND UNBOUNDED FOLLOWING) as summation16,
    SUM(c6) OVER(PARTITION BY c1 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 FOLLOWING) as summatio17,
    SUM(c3) OVER(PARTITION BY c1,c3 ROWS BETWEEN 3 PRECEDING AND UNBOUNDED FOLLOWING) as summation18,
    SUM(c6) OVER(PARTITION BY c1, c3, c6 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 FOLLOWING) as summation19,
    SUM(c2) OVER(PARTITION BY c5, c7, c9 ORDER BY c13, c5 ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as summation20,
    SUM(c2) OVER(PARTITION BY c5 ORDER BY c13, c5 ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as summation21,
    CORR(c2, c5) OVER(ROWS BETWEEN 3 PRECEDING AND 1 FOLLOWING) as corr1,
    CORR(c2, c5) OVER(ORDER BY c13 ROWS BETWEEN 3 PRECEDING AND 1 FOLLOWING) as corr2,
    CORR(c2, c3) OVER(ORDER BY c13 ROWS 3 PRECEDING) as corr3,
    CORR(c2, c3) OVER(ROWS BETWEEN 3 PRECEDING AND UNBOUNDED FOLLOWING) as corr4,
    CORR(c2, c3) OVER(ROWS BETWEEN UNBOUNDED PRECEDING AND 1 FOLLOWING) as corr5,
    CORR(c2, c3) OVER(ORDER BY c13 ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as corr6,
    CORR(c2, c3) OVER(ORDER BY c13 ROWS UNBOUNDED PRECEDING) as corr7,
    CORR(c2, c5) OVER(PARTITION BY c1 ROWS BETWEEN 3 PRECEDING AND UNBOUNDED FOLLOWING) as corr8,
    CORR(c4, c5) OVER(PARTITION BY c1 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 FOLLOWING) as corr9,
    CORR(c4, c9) OVER(PARTITION BY c1 ORDER BY c13 ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as corr10,
    CORR(c4, c11) OVER(PARTITION BY c2 ORDER BY c13 ROWS UNBOUNDED PRECEDING) as corr11,
    CORR(c4, c11) OVER(PARTITION BY c3 ROWS BETWEEN 3 PRECEDING AND UNBOUNDED FOLLOWING) as corr12,
    CORR(c8, c11) OVER(PARTITION BY c7 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 FOLLOWING) as corr13,
    CORR(c8, c11) OVER(PARTITION BY c13 ORDER BY c10 ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as corr14,
    CORR(c8, c11) OVER(PARTITION BY c13 ORDER BY c10 ROWS UNBOUNDED PRECEDING) as corr15
FROM test;
