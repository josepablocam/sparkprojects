/*
 A first cut draft in terms of what the KS test implementation might look like 
*/

package com.cloudera.sparkprojects
import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest

import org.apache.spark.rdd.RDD
import Math._

object KS {
  def ecdf(dat: RDD[Double]): RDD[Double] = {
    val n = dat.count
    dat.sortBy(x => x).zipWithIndex().map { case (v, i) => i / n }
  }

  // KS test for 1 set of sample data, versus theoretical, using string for name
  // of distribution
  def testOneSample(dat: RDD[Double], dist: String): (Double, Double) = {
    val cdf = 
      dist match {
        case "stdnorm" => (x: Double) => new NormalDistribution().cumulativeProbability(x)
        case  _ =>  throw new UnsupportedOperationException()
    }
    testOneSample(dat, cdf)
  }
  
  // KS test for 1 set of sample data (versus theoretical distribution)
  // 2-sided variant
  def testOneSample(dat: RDD[Double], cdf: Double => Double): (Double, Double) = {
    val empiriRDD = ecdf(dat) // empirical distribution
    val distances = empiriRDD.map { case (v, empVal) => Math.abs(cdf(v) -  empVal) }
    val ksStat = distances.max
    evalOneSided(ksStat, distances.length)
  }
  
  def testTwoSample(dat1: RDD[Double], dat2: RDD[Double]): (Double, Double) = { 
    if(dat1.count != dat2.count) {
      // throw exception for mismatched lengths (search mllib for existing exception)
      throw new Exception()
    }
    val empiri1 = ecdf(dat1)
    val empiri2 = ecdf(dat2)
    val distances = empiri1.zip(empiri2).map { case (p1, p2) => Math.abs(p1 - p2) }
    val ksStat = distances.max
    evalOneSided(ksStat, distances.length)
  }
  
  private[KS] def evalOneSided(ksstat: Double, n: Int): Double = {
    val pval = 1 - new KolmogorovSmirnovTest().cdf(ksStat, n)
    (ksStat, pval)
  }

}
