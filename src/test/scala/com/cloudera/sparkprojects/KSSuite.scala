/**
 * Copyright (c) 2015, Cloudera, Inc. All Rights Reserved.
 *
 * Cloudera, Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"). You may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * This software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for
 * the specific language governing permissions and limitations under the
 * License.
 */

package com.cloudera.sparkprojects

import org.apache.commons.math3.distribution.{ExponentialDistribution, NormalDistribution}
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest

import org.apache.spark.SparkContext

import org.scalatest.{FunSuite, ShouldMatchers}

class KSSuite extends FunSuite with ShouldMatchers {
  val sc = new SparkContext("local", "KSSuite")
  val stdNormDist = new NormalDistribution(0, 1)
  val expParam = 0.4
  val expDist = new ExponentialDistribution(expParam)

  test("ecdf") {
    val data = (1 to 100).toArray.map(_.toDouble)
    val distData = sc.parallelize(data)
    val dist = KS.ecdf(distData).collect()
    dist(0)._3 should be (0.01)
    dist(data.length - 1)._3 should be (1.0)
    dist(9)._3 should be (0.1)
  }

  test("1 sample") {
    val sampledNorm = sc.parallelize(stdNormDist.sample(1000))
    val sampledExp = sc.parallelize(expDist.sample(1000))
    val ksTest = new KolmogorovSmirnovTest()

    val threshold = 0.05

    val (stat1, pval1) = KS.testOneSample(sampledNorm, "stdnorm")
    stat1 should be (ksTest.kolmogorovSmirnovStatistic(stdNormDist, sampledNorm.collect()))
    pval1 should be > threshold // cannot reject H0

    val (stat2, pval2) = KS.testOneSample(sampledExp, "stdnorm")
    stat2 should be (ksTest.kolmogorovSmirnovStatistic(stdNormDist, sampledExp.collect()))
    pval2 should be < threshold // should reject H0

    // dist is not serializable, so will have to create in the lambda
    val expCDF = ((p: Double) => (x: Double) => {
      new ExponentialDistribution(p).cumulativeProbability(x)
    })(expParam)

    val (stat3, pval3) = KS.testOneSample(sampledExp, expCDF)
    stat3 should be (ksTest.kolmogorovSmirnovStatistic(expDist, sampledExp.collect()))
    pval3 should be > threshold // cannot reject H0

    val (stat4, pval4) = KS.testOneSample(sampledNorm, expCDF)
    stat4 should be (ksTest.kolmogorovSmirnovStatistic(expDist, sampledNorm.collect()))
    pval4 should be < threshold // should reject H0
  }
}
