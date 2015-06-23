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
    dist(0)._2 should be (0.01)
    dist(data.length - 1)._2 should be (1.0)
    dist(9)._2 should be (0.1)
  }

  test("1 sample") {
    val sampledNorm = sc.parallelize(stdNormDist.sample(1000))
    val sampledExp = sc.parallelize(expDist.sample(1000))

    val threshold = 0.05

    val accept1 = KS.testOneSample(sampledNorm, "stdnorm")
    accept1._2 should be > threshold // cannot reject H0

    val reject1 = KS.testOneSample(sampledExp, "stdnorm")
    reject1._2 should be < threshold // should reject H0

    // dist is not serializable, so will have to create in the lambda
    val expCDF = ((p: Double) => (x: Double) => {
      new ExponentialDistribution(p).cumulativeProbability(x)
      })(expParam)

    val accept2 = KS.testOneSample(sampledExp, expCDF)
    accept2._2 should be > threshold // cannot reject H0

    val reject2 = KS.testOneSample(sampledNorm, expCDF)
    reject2._2 should be < threshold // should reject H0
  }
}
