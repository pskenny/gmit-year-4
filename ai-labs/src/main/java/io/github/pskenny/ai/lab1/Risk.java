package io.github.pskenny.ai.lab1;

import net.sourceforge.jFuzzyLogic.FIS;
import net.sourceforge.jFuzzyLogic.plot.JFuzzyChart;

/*
 * Lab 1 from John Healy for Artificial Intelligence, (Honours) Software Development, GMIT
 * See: http://jfuzzylogic.sourceforge.net/html/manual.html#runfcl
 */
public class Risk {
	
	public Risk() {
		System.out.println(getRisk(60, 10));
	}

	public double getRisk(double funding, double staffing) {
		FIS fis = FIS.load("fcl/risk.fcl", true);

		// View fuzzy logic charts
		//JFuzzyChart.get().chart(fis);

		fis.setVariable("funding", funding);
		fis.setVariable("staffing", staffing);
		fis.evaluate();

		return fis.getVariable("risk").defuzzify();
	}
}
