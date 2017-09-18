starlings.I.model <- aov(Food.Intake~Gender*Day.Length,data=starlings)
summary(starlings.I.model)
model.tables(starlings.I.model)
contrasts(starlings$Day.Length)<-"contr.helmert"
contrasts(starlings$Gender)<-"contr.helmert"
coef(aov(Food.Intake~Gender*Day.Length,data=starlings))
