export interface ReportData {
    title: string;
    reportDate: string;
    metrics: {
        yesterdayBurn: string;
        predictedBurn: string;
        areaIncrease: string;
        percentIncrease: string;
        avgSpeed: string;
        direction: string;
    };
    hotspots: {
        x: number;
        y: number;
        confidence: number;
    }[];
    actionItems: string[];
    analysisImages: string[];
}
