# Naive bayes classifier
Брал инфу [отсюда](http://bazhenov.me/blog/2012/06/11/naive-bayes.html)

ROC-кривая для размытия 1 (хз че тут вообще получилось):

![ROC-кривая](https://user-images.githubusercontent.com/958656/32308253-f1471cee-bf95-11e7-944d-5feef254276e.png)

Тестовый прогон кросс-валидации, чтобы выбрать коэффицент размытия и порог при превышении которого письмо будет отправлено в спам:

```
cross-validation error: 61.47% | blur_k = 0 | margin = 0.1
cross-validation error: 61.56% | blur_k = 0 | margin = 0.2
cross-validation error: 61.65% | blur_k = 0 | margin = 0.3
cross-validation error: 61.65% | blur_k = 0 | margin = 0.4
cross-validation error: 61.93% | blur_k = 0 | margin = 0.5
cross-validation error: 62.02% | blur_k = 0 | margin = 0.6
cross-validation error: 62.02% | blur_k = 0 | margin = 0.7
cross-validation error: 62.02% | blur_k = 0 | margin = 0.8
cross-validation error: 61.83% | blur_k = 0 | margin = 0.9
cross-validation error: 64.31% | blur_k = 0 | margin = 1.0
cross-validation error: 4.22% | blur_k = 1 | margin = 0.1
cross-validation error: 4.22% | blur_k = 1 | margin = 0.2
cross-validation error: 4.13% | blur_k = 1 | margin = 0.3
cross-validation error: 4.04% | blur_k = 1 | margin = 0.4
cross-validation error: 4.04% | blur_k = 1 | margin = 0.5
cross-validation error: 3.94% | blur_k = 1 | margin = 0.6
cross-validation error: 3.85% | blur_k = 1 | margin = 0.7
cross-validation error: 3.76% | blur_k = 1 | margin = 0.8
cross-validation error: 3.76% | blur_k = 1 | margin = 0.9
cross-validation error: 7.16% | blur_k = 1 | margin = 1.0
cross-validation error: 5.14% | blur_k = 2 | margin = 0.1
cross-validation error: 5.05% | blur_k = 2 | margin = 0.2
cross-validation error: 5.05% | blur_k = 2 | margin = 0.3
cross-validation error: 4.95% | blur_k = 2 | margin = 0.4
cross-validation error: 4.95% | blur_k = 2 | margin = 0.5
cross-validation error: 4.77% | blur_k = 2 | margin = 0.6
cross-validation error: 4.68% | blur_k = 2 | margin = 0.7
cross-validation error: 4.59% | blur_k = 2 | margin = 0.8
cross-validation error: 4.50% | blur_k = 2 | margin = 0.9
cross-validation error: 7.80% | blur_k = 2 | margin = 1.0
cross-validation error: 6.15% | blur_k = 3 | margin = 0.1
cross-validation error: 6.06% | blur_k = 3 | margin = 0.2
cross-validation error: 5.96% | blur_k = 3 | margin = 0.3
cross-validation error: 5.78% | blur_k = 3 | margin = 0.4
cross-validation error: 5.78% | blur_k = 3 | margin = 0.5
cross-validation error: 5.78% | blur_k = 3 | margin = 0.6
cross-validation error: 5.60% | blur_k = 3 | margin = 0.7
cross-validation error: 5.50% | blur_k = 3 | margin = 0.8
cross-validation error: 5.41% | blur_k = 3 | margin = 0.9
cross-validation error: 8.35% | blur_k = 3 | margin = 1.0
cross-validation error: 6.24% | blur_k = 4 | margin = 0.1
cross-validation error: 6.24% | blur_k = 4 | margin = 0.2
cross-validation error: 6.24% | blur_k = 4 | margin = 0.3
cross-validation error: 6.24% | blur_k = 4 | margin = 0.4
cross-validation error: 6.15% | blur_k = 4 | margin = 0.5
cross-validation error: 6.15% | blur_k = 4 | margin = 0.6
cross-validation error: 6.06% | blur_k = 4 | margin = 0.7
cross-validation error: 5.96% | blur_k = 4 | margin = 0.8
cross-validation error: 5.96% | blur_k = 4 | margin = 0.9
cross-validation error: 8.62% | blur_k = 4 | margin = 1.0
cross-validation error: 6.51% | blur_k = 5 | margin = 0.1
cross-validation error: 6.51% | blur_k = 5 | margin = 0.2
cross-validation error: 6.42% | blur_k = 5 | margin = 0.3
cross-validation error: 6.42% | blur_k = 5 | margin = 0.4
cross-validation error: 6.42% | blur_k = 5 | margin = 0.5
cross-validation error: 6.42% | blur_k = 5 | margin = 0.6
cross-validation error: 6.33% | blur_k = 5 | margin = 0.7
cross-validation error: 6.24% | blur_k = 5 | margin = 0.8
cross-validation error: 6.06% | blur_k = 5 | margin = 0.9
cross-validation error: 8.99% | blur_k = 5 | margin = 1.0
cross-validation error: 7.06% | blur_k = 6 | margin = 0.1
cross-validation error: 6.79% | blur_k = 6 | margin = 0.2
cross-validation error: 6.88% | blur_k = 6 | margin = 0.3
cross-validation error: 6.88% | blur_k = 6 | margin = 0.4
cross-validation error: 6.88% | blur_k = 6 | margin = 0.5
cross-validation error: 6.88% | blur_k = 6 | margin = 0.6
cross-validation error: 6.79% | blur_k = 6 | margin = 0.7
cross-validation error: 6.61% | blur_k = 6 | margin = 0.8
cross-validation error: 6.42% | blur_k = 6 | margin = 0.9
cross-validation error: 9.08% | blur_k = 6 | margin = 1.0
cross-validation error: 7.43% | blur_k = 7 | margin = 0.1
cross-validation error: 7.52% | blur_k = 7 | margin = 0.2
cross-validation error: 7.16% | blur_k = 7 | margin = 0.3
cross-validation error: 6.97% | blur_k = 7 | margin = 0.4
cross-validation error: 6.88% | blur_k = 7 | margin = 0.5
cross-validation error: 6.88% | blur_k = 7 | margin = 0.6
cross-validation error: 6.88% | blur_k = 7 | margin = 0.7
cross-validation error: 6.88% | blur_k = 7 | margin = 0.8
cross-validation error: 6.70% | blur_k = 7 | margin = 0.9
cross-validation error: 9.82% | blur_k = 7 | margin = 1.0
cross-validation error: 7.71% | blur_k = 8 | margin = 0.1
cross-validation error: 7.52% | blur_k = 8 | margin = 0.2
cross-validation error: 7.52% | blur_k = 8 | margin = 0.3
cross-validation error: 7.52% | blur_k = 8 | margin = 0.4
cross-validation error: 7.34% | blur_k = 8 | margin = 0.5
cross-validation error: 7.06% | blur_k = 8 | margin = 0.6
cross-validation error: 6.97% | blur_k = 8 | margin = 0.7
cross-validation error: 6.97% | blur_k = 8 | margin = 0.8
cross-validation error: 6.79% | blur_k = 8 | margin = 0.9
cross-validation error: 9.82% | blur_k = 8 | margin = 1.0
cross-validation error: 7.89% | blur_k = 9 | margin = 0.1
cross-validation error: 7.80% | blur_k = 9 | margin = 0.2
cross-validation error: 7.61% | blur_k = 9 | margin = 0.3
cross-validation error: 7.61% | blur_k = 9 | margin = 0.4
cross-validation error: 7.61% | blur_k = 9 | margin = 0.5
cross-validation error: 7.52% | blur_k = 9 | margin = 0.6
cross-validation error: 7.43% | blur_k = 9 | margin = 0.7
cross-validation error: 7.06% | blur_k = 9 | margin = 0.8
cross-validation error: 6.88% | blur_k = 9 | margin = 0.9
cross-validation error: 10.00% | blur_k = 9 | margin = 1.0
cross-validation error: 7.89% | blur_k = 10 | margin = 0.1
cross-validation error: 7.89% | blur_k = 10 | margin = 0.2
cross-validation error: 7.89% | blur_k = 10 | margin = 0.3
cross-validation error: 7.71% | blur_k = 10 | margin = 0.4
cross-validation error: 7.61% | blur_k = 10 | margin = 0.5
cross-validation error: 7.71% | blur_k = 10 | margin = 0.6
cross-validation error: 7.61% | blur_k = 10 | margin = 0.7
cross-validation error: 7.52% | blur_k = 10 | margin = 0.8
cross-validation error: 7.06% | blur_k = 10 | margin = 0.9
cross-validation error: 10.18% | blur_k = 10 | margin = 1.0

minimal error: 3.76% | blur_k = 1 | margin = 0.8
```
