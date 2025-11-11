/*
All Types of Intervention-Methods have to be  registered here.
Provide the Name of the Intervention-Method used by the Flask-Backend (e.g. SAEIntervention)
and define an abbreviation for it to be used in the Frontend when displaying Features of a Layer (e.g. S26D123)
 */

export const toType= new Map<string, string> ([
    ["L", "LMDebuggerIntervention"]
]);

export const toAbbr = new Map(Array.from(toType, a => a.reverse() as [string, string]))

/*
Names of Intervention-Methods that are not sortable are defined here.
An Intervention-Method is not sortable, if the execution order of its Interventions doesn't matter.
e.g. LMDebuggerIntervention does only apply Hooks, which are always executed after weight-changing Methods
 => not sortable
 */

export const UNSORTABLE_METHODS = ["LMDebuggerIntervention"];
