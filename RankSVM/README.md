We directly use codes of the propensity-based rank-SVM[1].

The codes can be seen in http://www.cs.cornell.edu/people/tj/svm_light/svm_proprank.html

In this method, the inverse propensity weighting is added in input data.

The input data format is shown as following:

- 1 qid:1 cost:2.0 1:1 2:1 3:0 4:0.2 5:0 # 1A
- 0 qid:1 1:0 2:0 3:1 4:0.1 5:1 # 1B
- 0 qid:1 1:0 2:1 3:0 4:0.4 5:0 # 1C
- 0 qid:1 1:0 2:0 3:1 4:0.3 5:0 # 1D
- 1 qid:2 cost:3.3 1:1 2:0 3:1 4:0.4 5:0 # 2B
- 0 qid:2 1:0 2:0 3:1 4:0.2 5:0 # 2A
- 0 qid:2 1:0 2:0 3:1 4:0.1 5:0 # 2C
- 0 qid:2 1:0 2:0 3:1 4:0.2 5:0 # 2D
- 0 qid:2 1:0 2:0 3:1 4:0.1 5:1 # 2E
- 1 qid:3 cost:10.0 1:0 2:0 3:1 4:0.1 5:0 # 2C
- 0 qid:3 1:0 2:0 3:1 4:0.2 5:0 # 2A
- 0 qid:3 1:1 2:0 3:1 4:0.4 5:0 # 2B
- 0 qid:3 1:0 2:0 3:1 4:0.2 5:0 # 2D
- 0 qid:3 1:0 2:0 3:1 4:0.1 5:1 # 2E

When we get propenities of different biases from cxt-aware EM, we can calculate the Combinational propensity of every clicked sample based on its position and context information.  

Then put the inverse propensity value into th above "cost". For example, if 1A`s propensity is 0.8, the line of 1A in input data wil be: 
- 1 qid:1 cost:1.25 1:1 2:1 # 1A.


Reference:
[1]Joachims T , Swaminathan A , Schnabel T . Unbiased Learning-to-Rank with Biased Feedback[C]// Tenth Acm International Conference. ACM, 2017.
