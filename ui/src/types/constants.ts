/*
All Types of Intervention-Methods have to be  registered here.
Provide the Name of the Intervention-Method used by the Flask-Backend (e.g. SAEIntervention)
and define an abbreviation for it to be used in the Frontend when displaying Features of a Layer (e.g. S26D123)
 */

export const toType= new Map<string, string> ([
    ["L", "LMDebuggerIntervention"],
    ["S", "SAEIntervention"]
]);

export const toAbbr = new Map(Array.from(toType, a => a.reverse() as [string, string]))